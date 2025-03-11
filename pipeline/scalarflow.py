import json
import logging
import numpy as np
import os
import torch
import torchvision
from dataclasses import dataclass
from matplotlib.colors import to_rgb
from random import choice
from tqdm import tqdm

import arguments
import model
from arguments.__init__ import ModelParams
from dataset import DataLoader, DatasetInfo
from dataset.cameras import ShootModel
from dataset.readers import readScalarFlowInfo
from model.gaussfluids import Gaussfluids
from renderer import render
from renderer.network_tools import handle_network
from utils.general_utils import handle_factor
from utils.loss_utils import l1_loss, ssim, density_loss, aniso_loss, vol_loss, opacity_loss, feature_loss
from utils.math import ActivationType
from utils.sh_utils import rgb_str_to_sh_tensor
from utils.system_utils import dump_cfg
from utils.time_utils import TimeSeriesInfo


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
    dataloader = DataLoader(mdl.data_device, dataset, is_nerf_synthetic=False)
    if opt.end_frame == -1:
        opt.end_frame = dataloader.time_info.num_frames - 1

    gaussfluids = Gaussfluids(mdl.sh_degree, channel=1, base_time=dataloader.time_info.get_time(opt.end_frame),
                              hidden_sizes=mdl.hidden_sizes, track_channel=mdl.track_channel)

    if checkpoint:
        (model_params, first_iter, opt_dict) = torch.load(checkpoint)
        gaussfluids.restore(model_params)
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
        handle_network(pipe, gaussfluids, dataloader.time_info, background,
                       (iteration == int(opt.iterations)), source_path, opt.start_frame, opt.end_frame, opt.min_opacity)

        bg = torch.rand((1), device="cuda") if pipe.random_background else background

        if iteration <= opt.warm_iterations:
            frame_id = opt.end_frame
        elif iteration <= opt.dynamics_iterations:
            start_frame = int(opt.end_frame - (iteration / opt.dynamics_iterations) * (opt.end_frame - opt.start_frame))
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

        with torch.no_grad():
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

            elif iteration % 1000 == 0 and iteration != opt.iterations:
                gaussfluids.densify_and_prune(opt.densify_grad_threshold, opt.min_opacity,
                                              dataloader.cameras_extent, opt.max_screen_size)
                gaussfluids.split_ellipsoids(opt.target_radius, max_num=opt.max_num_points)

            # Optimizer step
            if iteration <= opt.iterations:
                gaussfluids.optimizer.step()
                gaussfluids.optimizer.zero_grad(set_to_none=True)

            if iteration % 1000 == 0:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussfluids.save(), iteration, gaussfluids.optimizer.state_dict()),
                           model_path + "/chkpnt" + str(iteration) + ".pth")


def rendering(source_path, output_path, mdl: ModelParams, opt: OptimizationParams, pipe: PipelineParams, checkpoint,
              scaling_factor=None, opacity_factor=None, bg_color=None, gs_color=None):
    dataset: DatasetInfo = readScalarFlowInfo(source_path, pipe.calib_folder)
    dataloader = DataLoader(mdl.data_device, dataset, shuffle=False, is_nerf_synthetic=False)
    if opt.end_frame == -1:
        opt.end_frame = dataloader.time_info.num_frames - 1
    gaussfluids = Gaussfluids(mdl.sh_degree, channel=1, base_time=dataloader.time_info.get_time(opt.end_frame),
                              hidden_sizes=mdl.hidden_sizes, track_channel=mdl.track_channel)
    (model_params, first_iter) = torch.load(checkpoint)
    gaussfluids.restore(model_params)

    shoot: ShootModel = dataloader.getTrainCameras()[0]
    logging.info(f"Using {shoot.shoot_info.image_name} as cam.")
    render_path = os.path.join(output_path, "renders")
    gts_path = os.path.join(output_path, "gt")
    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gts_path, exist_ok=True)

    bg_color = to_rgb(bg_color) if bg_color is not None else to_rgb("white") if mdl.white_background else to_rgb(
        "black")
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    gs_color_sh = rgb_str_to_sh_tensor(gs_color) if gs_color is not None else None

    for frame_index in range(dataloader.time_info.num_frames):
        print(frame_index)
        frame_time = dataloader.time_info.get_time(frame_index)
        with torch.no_grad():
            gaussians: model.Gaussians = gaussfluids.get_static(frame_time)
            if scaling_factor is not None:
                new_scaling = handle_factor(scaling_factor, gaussians.get_scaling)
                new_scaling = gaussians._inverse_scaling_activation(torch.min(new_scaling, torch.ones_like(
                    new_scaling)) if gaussians.scaling_activation_type == ActivationType.SIGMOID else new_scaling)
                gaussians.scaling = new_scaling.cuda()
            if opacity_factor is not None:
                new_opacity = handle_factor(opacity_factor, gaussians.get_opacity)
                new_opacity = gaussians._inverse_opacity_activation(torch.min(new_opacity, torch.ones_like(
                    new_opacity)) if gaussians.opacity_activation_type == ActivationType.SIGMOID else new_opacity)
                gaussians.opacity = new_opacity.cuda()
            if gs_color_sh is not None:
                if gs_color_sh.shape[0] == 3:
                    gaussians.channel = 3
                    gaussians.features_dc = gs_color_sh.unsqueeze(0).unsqueeze(0).tile(gaussians.get_num, 1, 1)
                    gaussians.features_rest = gaussians.features_rest.tile(1, 1, 3)
                elif gs_color_sh.shape[0] == 1:
                    gaussians.features_dc = gs_color_sh.unsqueeze(0).unsqueeze(0).tile(gaussians.get_num, 1, 1)
                else:
                    raise ValueError("gs_color_sh must be either 3 or 1 dimensional")
            rendering = render(shoot, gaussians, pipe, background)["render"]
            gt = shoot.image[0:3, :, :]
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:04d}'.format(frame_index) + ".png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:04d}'.format(frame_index) + ".png"))


def export_npz(source_path, output_path, mdl: ModelParams, opt: OptimizationParams, pipe: PipelineParams,
               checkpoint, time_info: TimeSeriesInfo = None, ply=True):
    with torch.no_grad():
        dataset: DatasetInfo = readScalarFlowInfo(source_path, pipe.calib_folder)
        dataloader = DataLoader(mdl.data_device, dataset, shuffle=False, is_nerf_synthetic=False)
        if opt.end_frame == -1:
            opt.end_frame = dataloader.time_info.num_frames - 1
        gaussfluids = Gaussfluids(mdl.sh_degree, channel=1, base_time=dataloader.time_info.get_time(opt.end_frame),
                                  hidden_sizes=mdl.hidden_sizes, track_channel=mdl.track_channel)
        (model_params, first_iter) = torch.load(checkpoint)
        gaussfluids.restore(model_params)

        time_info = dataloader.time_info if time_info is None else time_info

        save_path = os.path.join(output_path, 'npz')
        os.makedirs(save_path, exist_ok=True)

        json.dump(time_info._asdict(), open(os.path.join(save_path, 'time_info.json'), 'w'))

        for i in range(opt.start_frame, opt.end_frame + 1):
            time = time_info.start_time + i * time_info.time_step
            logging.info('Frame {}, Time {}'.format(i, time))
            gaussians = gaussfluids.get_static(time)
            gaussians.save_ply(os.path.join(save_path, 'ply', '{:04}.ply'.format(i))) if ply else None
            np.savez(os.path.join(save_path, '{:04}.npz'.format(i)), pos=gaussians.get_xyz.cpu().detach().numpy())
