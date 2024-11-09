#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import choice

from utils.loss_utils import l1_loss, ssim, density_loss, aniso_loss, vol_loss, opacity_loss, feature_loss
from renderer import render, network_gui
from dataset import DataLoader
from model import Gaussfluids, Gaussians
from trans_model import TransModel
from dataset.cameras import ShootModel
from utils.general_utils import safe_state
from tqdm import tqdm
from arguments.__init__ import ModelParams, PipelineParams, OptimizationParams
from renderer.network_tools import handle_network
from utils.system_utils import dump_cfg
from dataset.readers import readNeurofluidInfo

@dataclass
class Params:
    max_screen_size=1000

def training(mdl: ModelParams, opt: OptimizationParams, pipe, checkpoint):
    opt.bg_iterations = 0  # NeuroFluid dataset does not have bg

    first_iter = 0
    dump_cfg(mdl, model_path)
    scene_info = readNeurofluidInfo(source_path, mdl.white_background, eval=False,
                                    timestep_scaling=pipe.time_scaling)
    scene = DataLoader(mdl, scene_info)
    if opt.end_frame == -1:
        opt.end_frame = scene.time_info.num_frames - 1

    gaussians = Gaussfluids(mdl.sh_degree, base_time=scene.time_info.get_time(opt.end_frame),hidden_sizes=mdl.hidden_sizes,track_channel=mdl.track_channel)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        opt_dict = gaussians.restore(model_params)
        gaussians.setup(opt, scene.cameras_extent, position_lr_max_steps=opt.iterations - opt.dynamics_iterations,
                        opt_dict=opt_dict)
    else:
        gaussians.create_from_pcd(scene.point_cloud, init_color=pipe.dynamics_color)
        gaussians.setup(opt, scene.cameras_extent, position_lr_max_steps=opt.iterations - opt.dynamics_iterations)

    bg_color = [1, 1, 1] if mdl.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(0, opt.iterations), desc="Training progress", initial=first_iter)
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        handle_network(pipe, None, gaussians, scene.time_info, background,
                       (iteration == int(opt.iterations)), source_path, opt.start_frame, opt.end_frame, opt.min_opacity)
        iter_start.record()

        bg = torch.rand((3), device="cuda") if pipe.random_background else background

        if iteration <= opt.warm_iterations:
            frame_id = opt.end_frame
        elif iteration <= opt.dynamics_iterations:
            start_frame = int(opt.end_frame -
                              (iteration / opt.dynamics_iterations) * (opt.end_frame - opt.start_frame))
            frame_id = choice(range(start_frame, opt.end_frame + 1))
        else:
            gaussians.update_learning_rate(iteration - opt.dynamics_iterations)
            if iteration % 100 == 0:
                frame_id = opt.end_frame
            else:
                start_frame = opt.start_frame
                frame_id = choice(range(start_frame, opt.end_frame + 1))

        viewpoint_stack = scene.getTrainCameras(frame_index=frame_id)
        viewpoint_cam: ShootModel = choice(viewpoint_stack)
        gaussian_frame_dynamics: Gaussians = gaussians.get_static(viewpoint_cam.time)
        gaussian_frame = gaussian_frame_dynamics

        render_pkg = render(viewpoint_cam, gaussian_frame, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii, T_sum, T_count = render_pkg["render"], render_pkg[
            "viewspace_points"], \
            render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["T_sum"], render_pkg["T_count"]

        # Loss
        gt_image = viewpoint_cam.image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        if iteration <= opt.dynamics_iterations:
            loss = loss + opt.lambda_feats * feature_loss(gaussians.features_dc.squeeze(1))
        else:
            loss = loss + opt.lambda_dens * density_loss(gaussian_frame_dynamics.get_xyz)
            loss = loss + opt.lambda_aniso * aniso_loss(gaussian_frame_dynamics.get_scaling)
            loss = loss + opt.lambda_vol * vol_loss(gaussian_frame_dynamics.get_scaling)
            loss = loss + opt.lambda_opacity * opacity_loss(gaussians.get_opacity)
            loss = loss + opt.lambda_feats * feature_loss(gaussians.features_dc.squeeze(1), l=2)
        loss.backward()
        iter_end.record()

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
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter, T_sum.unsqueeze(-1),
                                                  T_count.unsqueeze(-1))

            if iteration <= opt.warm_iterations:
                if iteration % 1000 == 0 and iteration != opt.warm_iterations:
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.min_opacity,
                                                scene.cameras_extent, params.max_screen_size, prune_min_iters=200)
                    gaussians.split_ellipsoids(opt.target_radius, max_num=opt.max_num_points)
                    gaussians.reset_opacity()

                if iteration == opt.warm_iterations:
                    gaussians.prune_min_opacity(min_opacity=opt.min_opacity)

            elif iteration <= opt.dynamics_iterations:
                if iteration % 1000 == 0:
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.min_opacity,
                                                scene.cameras_extent, params.max_screen_size, prune_min_iters=200)
                    gaussians.split_ellipsoids(opt.target_radius, max_num=opt.max_num_points)
                    gaussians.double_scaling(multiplier=1.1)
                    gaussians.reset_opacity(gaussians.get_opacity.mean().cpu().detach().numpy())

            else:
                if iteration % 1000 == 0 and iteration != opt.iterations:
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.min_opacity,
                                                scene.cameras_extent, params.max_screen_size, prune_min_iters=200)
                    gaussians.split_ellipsoids(opt.target_radius, max_num=opt.max_num_points)
                    # gaussians.double_scaling()
                    # gaussians.reset_opacity(gaussians.get_opacity.mean())

            # Optimizer step
            if iteration <= opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration % 1000 == 0:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.save(), iteration),
                           model_path + "/chkpnt" + str(iteration) + ".pth")

def render_set(pp: PipelineParams, frame_index, shoot: ShootModel, background, render_path, gts_path, gaussfluids,
               frame_time, scaling_factor=None, opacity_factor=None, gs_color_sh:torch.Tensor=None):
    # mp, _ = ArgumentParser(neurofluid.ModelParams, allow_abbrev=False).parse_known_args()
    # pp, _ = ArgumentParser(neurofluid.PipelineParams, allow_abbrev=False).parse_known_args()
    # op, _ = ArgumentParser(neurofluid.OptimizationParams, allow_abbrev=False).parse_known_args()
    # dataset = readNeurofluidInfo(source_path, mp.white_background, eval=True, timestep_x=pp.time_scaling)

    # dataloader = DataLoader(mp, dataset, shuffle=False)
    # if mp.end_frame == -1:
    #     mp.end_frame = dataloader.time_info.num_frames - 1
    # gaussfluids = model.Gaussfluids(mp.sh_degree, base_time=dataloader.time_info.get_time(op.end_frame),
    #                           hidden_sizes=mp.hidden_sizes, track_channel=mp.track_channel)
    # (model_params, first_iter) = torch.load(args.start_checkpoint)
    # opt_dict = gaussfluids.restore(model_params)
    # shoot: ShootModel = dataloader.getTrainCameras()[0]

    # render_path = os.path.join(model_path, "renders")
    # gts_path = os.path.join(model_path, "gt")
    # os.makedirs(render_path, exist_ok=True)
    # os.makedirs(gts_path, exist_ok=True)
    #
    # bg_color = [1, 1, 1] if mp.white_background else [0, 0, 0]
    # background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # for frame_index in range(dataloader.time_info.num_frames):
    #     print(frame_index)
    #     frame_time = dataloader.time_info.get_time(frame_index)
    #     render_set(pp, frame_index, background=background, render_path=render_path, gts_path=gts_path,
    #                gaussfluids=gaussfluids, shoot=shoot, frame_time=frame_time)

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
        rendering = render(shoot, gaussians, pp, background)["render"]
        gt = shoot.image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:04d}'.format(frame_index) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:04d}'.format(frame_index) + ".png"))

def dump_sets(mdl: ModelParams, opt: OptimizationParams, pipe, checkpoint, time_info: TimeSeriesInfo = None):
    with torch.no_grad():
        dataset = readNeurofluidInfo(source_path, mdl.white_background, eval=False,
                                        timestep_scaling=pipe.time_scaling)
        dataloader = DataLoader(mdl, dataset)
        if opt.end_frame == -1:
            opt.end_frame = dataloader.time_info.num_frames - 1
        gaussians = Gaussfluids(mdl.sh_degree, base_time=dataloader.time_info.get_time(opt.end_frame), hidden_sizes=mdl.hidden_sizes,track_channel=mdl.track_channel)
        if checkpoint:
            (model_params, first_iter) = torch.load(checkpoint)
            opt_dict = gaussians.restore(model_params)

        else:
            raise Exception("No chkpnt specify")

        if time_info is None:
            time_info = dataloader.time_info

        bg_color = [1, 1, 1] if mdl.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        save_path = os.path.join(model_path, 'npz')
        os.makedirs(save_path, exist_ok=True)

        json.dump(time_info._asdict(), open(os.path.join(save_path, 'time_info.json'), 'w'))

        for i in range(opt.start_frame, opt.end_frame + 1):
            handle_network(pipe, None, gaussians, time_info, background, (i == opt.end_frame),
                           source_path, opt.start_frame, opt.end_frame, opt.min_opacity)
            time = time_info.start_time + i * time_info.time_step
            print('Frame {}, Time {}'.format(i, time))
            gaussian_frame = gaussians.get_static(time)
            gaussian_frame.save_ply(os.path.join(save_path, 'ply', '{:04}.ply'.format(i)))
            np.savez(os.path.join(save_path, '{:04}.npz'.format(i)), pos=gaussian_frame.get_xyz.cpu().detach().numpy())

def filter_gaussian(gaussian_frame: GaussfluidsModel):
    xyz = gaussian_frame.get_xyz.cpu().numpy()
    mask = gaussian_frame.get_opacity.cpu().numpy() < 0.1
    xyz_filtered = xyz[mask[:, 0]]
    return xyz_filtered

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--ip', type=str, default="0.0.0.0")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args()
    lp = ArgumentParser(ModelParams).parse_args()
    pp = ArgumentParser(PipelineParams).parse_args()
    op = ArgumentParser(OptimizationParams).parse_args()

    os.makedirs(args.model_path, exist_ok=True)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(silent=False)
    # Start GUI server, configure
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(mode=False)

    # run training
    training(lp, op, pp, args.start_checkpoint)
    # All done
    print("\nTraining complete.")
