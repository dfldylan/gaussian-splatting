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
import yaml
from random import choice

from utils.loss_utils import l1_loss, ssim, density_loss, aniso_loss, vol_loss, opacity_loss, feature_loss, \
    position_loss
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from scene.trans_model import TransModel
from scene.cameras import Camera
from utils.general_utils import safe_state
from tqdm import tqdm
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.sh_utils import RGB2SH
from gaussian_renderer.network_tools import handle_network
from utils.system_utils import merge_args, prepare_output_and_logger


def training(dataset: ModelParams, opt: OptimizationParams, pipe, checkpoint, fluid_setup):
    yaml_conf = yaml.safe_load(open(os.path.join(dataset.source_path, 'fluid.yml')))
    merge_args(dataset, opt, yaml_conf, fluid_setup)

    first_iter = 0
    _ = prepare_output_and_logger(dataset)
    scene = Scene(dataset)
    if opt.end_frame == -1:
        opt.end_frame = scene.time_info.num_frames - 1

    gs_bg = GaussianModel(dataset.sh_degree)
    gaussians = GaussianModel(dataset.sh_degree)
    trans = TransModel(dataset, scene.time_info, opt.end_frame)

    if checkpoint:
        (bg_params, trans_params, first_iter, model_params) = torch.load(checkpoint)
        gs_bg.restore(bg_params, opt, position_lr_max_steps=opt.bg_iterations)
        gaussians.restore(model_params, opt, position_lr_max_steps=opt.iterations - opt.dynamics_iterations)
        trans.restore(trans_params, opt, reset_time=False)
    else:
        gs_bg.create_from_pcd(scene.point_cloud, scene.cameras_extent)
        gs_bg.training_setup(opt, position_lr_max_steps=opt.bg_iterations)
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
        handle_network(pipe, gs_bg, gaussians, trans, scene.time_info, background,
                       (iteration == int(opt.iterations)), dataset, opt)
        iter_start.record()

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        if iteration <= opt.bg_iterations:
            # Every 1000 its we increase the levels of SH up to a maximum degree
            gs_bg.update_learning_rate(iteration)
            if iteration <= opt.bg_iterations * 0.3:
                viewpoint_stack = scene.getTrainCameras(frame_index=0)
            else:
                if iteration % 1000 == 0:
                    gs_bg.oneupSHdegree()
                frame_id = choice(range(opt.static_start, opt.static_end + 1))
                viewpoint_stack = scene.getTrainCameras(frame_index=frame_id)
            viewpoint_cam: Camera = choice(viewpoint_stack)
            gaussian_frame = gs_bg.move_0()
        else:
            # ensure only feature is require_grad on gaussian_0
            gs_bg.fixed_pose()
            gs_bg.fixed_feature_rest()
            gs_bg.fixed_feature_dc()
            gaussian_frame = gs_bg.move_0()
            gaussians.fixed_feature_rest()

            if iteration - opt.bg_iterations <= opt.warm_iterations:
                frame_id = opt.end_frame
            elif iteration <= opt.dynamics_iterations:
                start_frame = int(opt.end_frame - (iteration / opt.dynamics_iterations) * (
                        opt.end_frame - opt.start_frame))
                frame_id = choice(range(start_frame, opt.end_frame + 1))
            else:
                gaussians.update_learning_rate(iteration - opt.dynamics_iterations)
                if iteration % 100 == 0:
                    frame_id = opt.end_frame
                else:
                    start_frame = opt.start_frame
                    frame_id = choice(range(start_frame, opt.end_frame + 1))

            viewpoint_stack = scene.getTrainCameras(frame_index=frame_id)
            viewpoint_cam: Camera = choice(viewpoint_stack)
            dt_xyz, dt_scaling, dt_rotation = trans(viewpoint_cam.time)
            gaussian_frame_dynamics = gaussians.move(dt_xyz, dt_scaling, dt_rotation)
            gaussian_frame.add_extra_gaussians(gaussian_frame_dynamics)

        render_pkg = render(viewpoint_cam, gaussian_frame, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii, T_sum, T_count = render_pkg["render"], render_pkg[
            "viewspace_points"], \
            render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["T_sum"], render_pkg["T_count"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        if iteration > opt.bg_iterations:
            if iteration <= opt.bg_iterations + opt.warm_iterations:
                loss = (loss + opt.lambda_feats * feature_loss(gaussians._features_dc.squeeze(1),
                                                               mean=RGB2SH(
                                                                   torch.Tensor(dataset.dynamics_color).cuda())))
            elif iteration <= opt.dynamics_iterations:
                loss = (loss + opt.lambda_feats * feature_loss(gaussians._features_dc.squeeze(1),
                                                               mean=RGB2SH(
                                                                   torch.Tensor(dataset.dynamics_color).cuda())))
                loss = loss + 0.01 * position_loss(gaussians.get_xyz)
            else:
                loss = (loss + opt.lambda_feats * feature_loss(gaussians._features_dc.squeeze(1), mean=RGB2SH(
                    torch.Tensor(dataset.dynamics_color).cuda()), l=2))
                # loss = loss + opt.lambda_opacity * opacity_loss(gaussians.get_opacity)
                # loss = loss + opt.lambda_aniso * aniso_loss(gaussian_frame_dynamics.get_scaling)
                # loss = loss + 0.5 * opt.lambda_vol * vol_loss(gaussian_frame_dynamics.get_scaling)
                loss = loss + opt.lambda_dens * density_loss(gaussian_frame_dynamics.get_xyz, k=16)
                loss = loss + 0.01 * position_loss(gaussian_frame_dynamics.get_xyz, l=1)

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

            if iteration <= opt.bg_iterations:
                # Densification
                if iteration <= opt.bg_iterations * 0.8:
                    # Keep track of max radii in image-space for pruning
                    gs_bg.max_radii2D[visibility_filter] = torch.max(gs_bg.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])
                    gs_bg.add_densification_stats(viewspace_point_tensor.grad, visibility_filter)

                    if iteration > 500 and iteration % 100 == 0:
                        gs_bg.densify_and_prune(opt.densify_grad_threshold, opt.min_opacity, scene.cameras_extent,
                                                prune_min_iters=30)

                    if iteration % 2_000 == 0 or (dataset.white_background and iteration == 500):
                        gs_bg.reset_opacity()

            else:
                # Keep track of max radii in image-space for pruning
                visibility_filter = visibility_filter[-gaussians.get_num:]
                radii = radii[-gaussians.get_num:]
                viewspace_point_tensor_grad = viewspace_point_tensor.grad[-gaussians.get_num:]
                T_sum = T_sum[-gaussians.get_num:].unsqueeze(-1)
                T_count = T_count[-gaussians.get_num:].unsqueeze(-1)
                if visibility_filter.sum().cpu().numpy() != 0:
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                         radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter, T_sum, T_count)

                dynamics_iter = iteration - opt.bg_iterations
                if dynamics_iter <= opt.warm_iterations:
                    if dynamics_iter == 1000:
                        gaussians.prune_min_opacity(opt.min_opacity, trans=trans)
                        gaussians.prune_district(eps=opt.eps, min_samples=10, first_class=dataset.first_class,
                                                 trans=trans)

                    if dynamics_iter % 1000 == 0 and dynamics_iter != opt.warm_iterations:
                        gaussians.densify_and_prune(opt.densify_grad_threshold, opt.min_opacity, scene.cameras_extent,
                                                    1000, prune_min_iters=200, prune_min_T=0.1, trans=trans)
                        gaussians.split_ellipsoids(dataset.target_radius, max_num=opt.max_num_points, trans=trans)
                        gaussians.reset_opacity()
                        if dynamics_iter % 2000 == 0:
                            gs_bg.reset_opacity()

                    if dynamics_iter == opt.warm_iterations:
                        gaussians.prune_min_opacity(opt.min_opacity, trans=trans)

                elif iteration <= opt.dynamics_iterations:
                    if iteration % 1000 == 0:
                        gaussians.densify_and_prune(opt.densify_grad_threshold, opt.min_opacity, scene.cameras_extent,
                                                    None, prune_min_iters=500, prune_min_T=0.1, trans=trans)
                        gaussians.split_ellipsoids(dataset.target_radius, max_num=opt.max_num_points, trans=trans)
                        gaussians.double_scaling(multiplier=1.1)
                        if iteration % 5000 == 0:
                            gaussians.reset_opacity()
                            gs_bg.reset_opacity()


                else:
                    if iteration % 1000 == 0 and iteration != opt.iterations:
                        gaussians.prune_points((torch.vstack([gaussians.get_scaling[:,(0,2)].prod(1),gaussians.get_scaling[:,(0,1)].prod(1),gaussians.get_scaling[:,(1,2)].prod(1)])>1e-2).any(0), trans=trans)
                        gaussians.densify_and_prune(opt.densify_grad_threshold, opt.min_opacity,
                                                    scene.cameras_extent, 5000, prune_min_iters=500, prune_min_T=0.005,
                                                    trans=trans)
                        gaussians.split_ellipsoids(dataset.target_radius, max_num=opt.max_num_points, trans=trans)
                        # gaussians.double_scaling(multiplier=1.1)
                        if iteration % 5000 == 0:
                            gaussians.reset_opacity()
                            gs_bg.reset_opacity()

            # Optimizer step
            if iteration <= opt.iterations:
                gs_bg.optimizer.step()
                gs_bg.optimizer.zero_grad(set_to_none=True)
                gaussians.optimizer.step()
                trans.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                trans.optimizer.zero_grad(set_to_none=True)

            if iteration % 1000 == 0:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gs_bg.capture(), trans.capture(), iteration, gaussians.capture()),
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
    parser.add_argument('--fluid_setup', type=int, default=1)
    args = parser.parse_args(sys.argv[1:])

    os.makedirs(args.model_path, exist_ok=True)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.start_checkpoint, args.fluid_setup)

    # All done
    print("\nTraining complete.")
