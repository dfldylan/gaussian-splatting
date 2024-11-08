import logging
import os.path
from dataclasses import dataclass
from random import choice

import torch
from tqdm import tqdm

import arguments
from arguments.__init__ import ModelParams
from dataset import DataLoader, DatasetInfo
from dataset.cameras import ShootModel
from dataset.readers import readScalarFlowInfo
from model import Gaussfluids
from renderer import render
from renderer.network_tools import handle_network
from utils.loss_utils import l1_loss, ssim, density_loss, aniso_loss, vol_loss, opacity_loss, feature_loss
from utils.system_utils import dump_cfg


@dataclass
class PipelineParams(arguments.PipelineParams):
    calib_folder: str = ""


@dataclass
class OptimizationParams(arguments.OptimizationParams):
    max_screen_size: int = 1000


def training(source_path, model_path, mdl: ModelParams, opt: OptimizationParams, pipe: PipelineParams, checkpoint=None):
    first_iter = 0
    dump_cfg(mdl, model_path)
    dataset: DatasetInfo = readScalarFlowInfo(source_path, pipe.calib_folder)
    dataloader = DataLoader(mdl.data_device, dataset)
    if opt.end_frame == -1:
        opt.end_frame = dataloader.time_info.num_frames - 1

    gaussfluids = Gaussfluids(mdl.sh_degree, channel=1, base_time=dataloader.time_info.get_time(opt.end_frame),
                              hidden_sizes=mdl.hidden_sizes, track_channel=mdl.track_channel)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        opt_dict = gaussfluids.restore(model_params)
        gaussfluids.setup(opt, dataloader.cameras_extent, position_lr_max_steps=opt.iterations, opt_dict=opt_dict)
    else:
        gaussfluids.create_from_pcd(dataloader.point_cloud, init_color=pipe.dynamics_color)
        gaussfluids.setup(opt, dataloader.cameras_extent, position_lr_max_steps=opt.iterations)

    bg_color = [1, 1, 1] if mdl.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(0, opt.iterations), desc="Training progress", initial=first_iter)
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        handle_network(pipe, None, gaussfluids, dataloader.time_info, background,
                       (iteration == int(opt.iterations)), source_path, opt.start_frame, opt.end_frame, opt.min_opacity)

        bg = torch.rand((1), device="cuda") if pipe.random_background else background

        if iteration <= opt.warm_iterations:
            frame_id = opt.end_frame
        elif iteration <= opt.dynamics_iterations:
            start_frame = int(opt.end_frame -
                              (iteration / opt.dynamics_iterations) * (opt.end_frame - opt.start_frame))
            frame_id = choice(range(start_frame, opt.end_frame + 1))
        else:
            gaussfluids.update_learning_rate(iteration - opt.dynamics_iterations)
            if iteration % 100 == 0:
                frame_id = opt.end_frame
            else:
                start_frame = opt.start_frame
                frame_id = choice(range(start_frame, opt.end_frame + 1))

        shoot_stack = dataloader.getTrainCameras(frame_index=frame_id)
        shoot: ShootModel = choice(shoot_stack)
        gaussians = gaussfluids.get_static(shoot.time)

        render_pkg = render(shoot, gaussians, pipe, bg)
        image, viewspace_point_tensor = render_pkg["render"], render_pkg["viewspace_points"]
        visibility_filter, radii = render_pkg["visibility_filter"], render_pkg["radii"]
        T_sum, T_count = render_pkg["T_sum"], render_pkg["T_count"]

        # Loss
        gt_image = shoot.image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        if iteration <= opt.dynamics_iterations:
            loss = loss + opt.lambda_feats * feature_loss(gaussfluids.features_dc.squeeze(1))
        else:
            loss = loss + opt.lambda_dens * density_loss(gaussians.get_xyz)
            loss = loss + opt.lambda_aniso * aniso_loss(gaussians.get_scaling)
            loss = loss + opt.lambda_vol * vol_loss(gaussians.get_scaling)
            loss = loss + opt.lambda_opacity * opacity_loss(gaussfluids.get_opacity)
            loss = loss + opt.lambda_feats * feature_loss(gaussfluids.features_dc.squeeze(1), l=2)
        loss.backward()

        with (torch.no_grad()):
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 100 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(100)
            if iteration == opt.iterations:
                progress_bar.close()

            # Keep track of max radii in image-space for pruning
            viewspace_point_tensor_grad = viewspace_point_tensor.grad
            if visibility_filter.sum().cpu().numpy() != 0:
                gaussfluids.max_radii2D[visibility_filter] = torch.max(gaussfluids.max_radii2D[visibility_filter],
                                                                       radii[visibility_filter])
                gaussfluids.add_densification_stats(viewspace_point_tensor_grad, visibility_filter, T_sum.unsqueeze(-1),
                                                    T_count.unsqueeze(-1))

            if iteration <= opt.warm_iterations:
                if iteration % 500 == 0 and iteration != opt.warm_iterations:
                    gaussfluids.densify_and_prune(opt.densify_grad_threshold, opt.min_opacity,
                                                  dataloader.cameras_extent, opt.max_screen_size)
                    gaussfluids.split_ellipsoids(opt.target_radius, max_num=opt.max_num_points)
                    gaussfluids.reset_opacity(value=2 * opt.min_opacity)

                if iteration == opt.warm_iterations:
                    gaussfluids.prune_min_opacity(min_opacity=opt.min_opacity)

            elif iteration <= opt.dynamics_iterations:
                if iteration % 1000 == 0:
                    gaussfluids.densify_and_prune(opt.densify_grad_threshold, opt.min_opacity,
                                                  dataloader.cameras_extent, opt.max_screen_size)
                    gaussfluids.split_ellipsoids(opt.target_radius, max_num=opt.max_num_points)
                    gaussfluids.double_scaling(multiplier=1.1)
                    gaussfluids.reset_opacity(gaussfluids.get_opacity.mean().cpu().detach().numpy())

            else:
                if iteration % 1000 == 0 and iteration != opt.iterations:
                    gaussfluids.densify_and_prune(opt.densify_grad_threshold, opt.min_opacity,
                                                  dataloader.cameras_extent, opt.max_screen_size)
                    gaussfluids.split_ellipsoids(opt.target_radius, max_num=opt.max_num_points)

            # Optimizer step
            if iteration <= opt.iterations:
                gaussfluids.optimizer.step()
                gaussfluids.optimizer.zero_grad(set_to_none=True)

            if iteration % 1000 == 0:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussfluids.save(), iteration), model_path + "/chkpnt" + str(iteration) + ".pth")


from render import render_set


def rendering(source_path, output_path, mdl: ModelParams, opt: OptimizationParams, pipe: PipelineParams, checkpoint,
              scaling_factor=None, opacity_factor=None):
    dataset: DatasetInfo = readScalarFlowInfo(source_path, pipe.calib_folder)
    dataloader = DataLoader(mdl.data_device, dataset, shuffle=False)
    if opt.end_frame == -1:
        opt.end_frame = dataloader.time_info.num_frames - 1
    gaussfluids = Gaussfluids(mdl.sh_degree, channel=1, base_time=dataloader.time_info.get_time(opt.end_frame),
                              hidden_sizes=mdl.hidden_sizes, track_channel=mdl.track_channel)
    (model_params, first_iter) = torch.load(checkpoint)
    opt_dict = gaussfluids.restore(model_params)
    gaussfluids.setup(opt, dataloader.cameras_extent, position_lr_max_steps=opt.iterations, opt_dict=opt_dict)

    shoot: ShootModel = dataloader.getTrainCameras()[0]
    logging.info(f"Using {shoot.shoot_info.image_name} as cam.")
    render_path = os.path.join(output_path, "renders")
    gts_path = os.path.join(output_path, "gt")
    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gts_path, exist_ok=True)

    bg_color = [1, 1, 1] if mdl.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    for frame_index in range(dataloader.time_info.num_frames):
        print(frame_index)
        frame_time = dataloader.time_info.get_time(frame_index)
        render_set(pipe, frame_index, background=background, render_path=render_path, gts_path=gts_path,
                   gaussfluids=gaussfluids, shoot=shoot, frame_time=frame_time, scaling_factor=scaling_factor,
                   opacity_factor=opacity_factor)
