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
import sys
import torch
from random import choice

from utils.loss_utils import l1_loss, ssim, density_loss, aniso_loss, vol_loss, opacity_loss, feature_loss
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from scene.trans_model import TransModel
from scene.cameras import Camera
from utils.general_utils import safe_state
from tqdm import tqdm
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer.network_tools import handle_network
from utils.system_utils import prepare_output_and_logger


def training(dataset: ModelParams, opt: OptimizationParams, pipe, checkpoint):
    opt.bg_iterations = 0  # NeuroFluid dataset does not have bg

    first_iter = 0
    _ = prepare_output_and_logger(dataset)
    scene = Scene(dataset)
    if dataset.end_frame == -1:
        dataset.end_frame = scene.time_info.num_frames - 1

    gaussians = GaussianModel(dataset.sh_degree)
    trans = TransModel(dataset, scene.time_info)

    if checkpoint:
        (model_params, trans_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt, position_lr_max_steps=opt.iterations - opt.dynamics_iterations)
        trans.restore(trans_params, opt, reset_time=False)
    else:
        gaussians.create_from_pcd(scene.point_cloud, scene.cameras_extent, init_color=dataset.dynamics_color)
        gaussians.training_setup(opt, position_lr_max_steps=opt.iterations - opt.dynamics_iterations)
        trans.set_model(dataset, gaussians.get_num, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(0, opt.iterations), desc="Training progress", initial=first_iter)
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        handle_network(pipe, None, gaussians, trans, scene.time_info, background,
                       (iteration == int(opt.iterations)), dataset, opt.min_opacity)
        iter_start.record()

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        if iteration <= opt.warm_iterations:
            frame_id = dataset.end_frame
        elif iteration <= opt.dynamics_iterations:
            start_frame = int(dataset.end_frame -
                              (iteration / opt.dynamics_iterations) * (dataset.end_frame - dataset.start_frame))
            frame_id = choice(range(start_frame, dataset.end_frame + 1))
        else:
            gaussians.update_learning_rate(iteration - opt.dynamics_iterations)
            start_frame = dataset.start_frame
            frame_id = choice(range(start_frame, dataset.end_frame + 1))

        viewpoint_stack = scene.getTrainCameras(frame_index=frame_id)
        viewpoint_cam: Camera = choice(viewpoint_stack)
        dt_xyz, dt_scaling, dt_rotation = trans(viewpoint_cam.time)
        gaussian_frame_dynamics = gaussians.move(dt_xyz, dt_scaling, dt_rotation)
        gaussian_frame = gaussian_frame_dynamics

        render_pkg = render(viewpoint_cam, gaussian_frame, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
            render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss = loss + opt.lambda_dens * density_loss(gaussian_frame_dynamics.get_xyz)
        loss = loss + opt.lambda_aniso * aniso_loss(gaussian_frame_dynamics.get_scaling)
        loss = loss + opt.lambda_vol * vol_loss(gaussian_frame_dynamics.get_scaling)
        loss = loss + opt.lambda_opacity * opacity_loss(gaussians.get_opacity)
        loss = loss + opt.lambda_feats * feature_loss(gaussians._features_dc.squeeze(1))
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
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

            if iteration <= opt.warm_iterations:
                if iteration % 1000 == 0 and iteration != opt.warm_iterations:
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.min_opacity,
                                                scene.cameras_extent, 1000, prune_min_iters=200, trans=trans)
                    gaussians.split_ellipsoids(dataset.target_radius, max_num=opt.max_num_points, trans=trans)
                    gaussians.reset_opacity()

                if iteration == opt.warm_iterations:
                    gaussians.prune_min_opacity(min_opacity=opt.min_opacity, trans=trans)

            elif iteration <= opt.dynamics_iterations:
                if iteration % 1000 == 0:
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.min_opacity,
                                                scene.cameras_extent, 1000, prune_min_iters=200, trans=trans)
                    gaussians.split_ellipsoids(dataset.target_radius, max_num=opt.max_num_points, trans=trans)
                    gaussians.double_scaling()
                    gaussians.reset_opacity(gaussians.get_opacity.mean())

            else:
                if iteration % 1000 == 0 and iteration != opt.iterations:
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.min_opacity,
                                                scene.cameras_extent, 1000, prune_min_iters=200, trans=trans)
                    gaussians.split_ellipsoids(dataset.target_radius, max_num=opt.max_num_points, trans=trans)
                    # gaussians.double_scaling()
                    # gaussians.reset_opacity(gaussians.get_opacity.mean())

            # Optimizer step
            if iteration <= opt.iterations:
                gaussians.optimizer.step()
                trans.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                trans.optimizer.zero_grad(set_to_none=True)

            if iteration % 1000 == 0:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), trans.capture(), iteration),
                           scene.model_path + "/chkpnt" + str(iteration) + ".pth")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="0.0.0.0")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument('--fluid_setup', type=int, default=0)
    args = parser.parse_args(sys.argv[1:])

    os.makedirs(args.model_path, exist_ok=True)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.start_checkpoint)

    # All done
    print("\nTraining complete.")
