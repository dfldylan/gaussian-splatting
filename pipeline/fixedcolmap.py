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
from argparse import ArgumentParser
from random import choice

import torch
import yaml
from tqdm import tqdm

from arguments.__init__ import ModelParams, PipelineParams, OptimizationParams
from dataset import DataLoader
from dataset.cameras import ShootModel
from dataset.readers import readFixedColmapInfo
from model import Gaussfluids
from renderer import render, network_gui
from renderer.network_tools import handle_network
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, ssim, density_loss, feature_loss, \
    position_loss
from utils.sh_utils import RGB2SH, rgb_str_to_tensor
from utils.system_utils import merge_args, dump_cfg


@dataclass
class Params:
    max_screen_size: int = 5000
    first_class: int = 0


def training(mdl: ModelParams, opt: OptimizationParams, pipe, checkpoint, fluid_setup):
    yaml_conf = yaml.safe_load(open(os.path.join(source_path, 'fluid.yml')))
    merge_args(mdl, opt, yaml_conf, fluid_setup)

    first_iter = 0
    dump_cfg(mdl, model_path)
    scene_info = readFixedColmapInfo(source_path, eval=False)
    scene = DataLoader(mdl, scene_info,is_nerf_synthetic=False)
    if opt.end_frame == -1:
        opt.end_frame = scene.time_info.num_frames - 1

    gs_bg = Gaussfluids(sh_degree=3)
    gaussians = Gaussfluids(mdl.sh_degree, base_time=scene.time_info.get_time(opt.end_frame),
                            hidden_sizes=mdl.hidden_sizes, track_channel=mdl.track_channel)

    if checkpoint:
        (bg_params, first_iter, model_params) = torch.load(checkpoint)
        opt_dict = gs_bg.restore(bg_params, position_lr_max_steps=opt.bg_iterations)
        opt_dict = gaussians.restore(model_params)
    else:
        gs_bg.create_from_pcd(scene.point_cloud)
        gs_bg.setup(opt, scene.cameras_extent, position_lr_max_steps=opt.bg_iterations)
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
        handle_network(pipe, gaussians, scene.time_info, background,
                       (iteration == int(opt.iterations)), source_path, opt.start_frame, opt.end_frame, opt.min_opacity, gs_bg)
        iter_start.record()

        bg = torch.rand((3), device="cuda") if pipe.random_background else background

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
            viewpoint_cam: ShootModel = choice(viewpoint_stack)
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
            viewpoint_cam: ShootModel = choice(viewpoint_stack)
            gaussian_frame_dynamics = gaussians.get_static(viewpoint_cam.time)
            gaussian_frame.add_gaussians(gaussian_frame_dynamics)

        render_pkg = render(viewpoint_cam, gaussian_frame, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii, T_sum, T_count = render_pkg["render"], render_pkg[
            "viewspace_points"], \
            render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["T_sum"], render_pkg["T_count"]

        # Loss
        gt_image = viewpoint_cam.image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        if iteration > opt.bg_iterations:
            if iteration <= opt.bg_iterations + opt.warm_iterations:
                loss = (loss + opt.lambda_feats * feature_loss(gaussians.features_dc.squeeze(1),
                                                               mean=RGB2SH(rgb_str_to_tensor(init_color))))
            elif iteration <= opt.dynamics_iterations:
                loss = (loss + opt.lambda_feats * feature_loss(gaussians.features_dc.squeeze(1),
                                                               mean=RGB2SH(rgb_str_to_tensor(init_color))))
                loss = loss + 0.01 * position_loss(gaussians.get_xyz)
            else:
                loss = (loss + opt.lambda_feats * feature_loss(gaussians.features_dc.squeeze(1),
                                                               mean=RGB2SH(rgb_str_to_tensor(init_color)), l=2))
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
                                                prune_min_iters=10)

                    if iteration % 2_000 == 0 or (mdl.white_background and iteration == 500):
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
                        gaussians.prune_min_opacity(opt.min_opacity)
                        gaussians.prune_district(eps=opt.eps, min_samples=10, first_class=params.first_class)

                    if dynamics_iter % 1000 == 0 and dynamics_iter != opt.warm_iterations:
                        gaussians.densify_and_prune(opt.densify_grad_threshold, opt.min_opacity, scene.cameras_extent,
                                                    None, prune_min_iters=200, prune_min_T=0.1)
                        gaussians.split_ellipsoids(opt.target_radius, max_num=opt.max_num_points)
                        gaussians.reset_opacity()
                        if dynamics_iter % 2000 == 0:
                            gs_bg.reset_opacity()

                    if dynamics_iter == opt.warm_iterations:
                        gaussians.prune_min_opacity(opt.min_opacity)

                elif iteration <= opt.dynamics_iterations:
                    if iteration % 1000 == 0:
                        gaussians.prune_points((torch.vstack(
                            [gaussians.get_scaling[:, (0, 2)].prod(1), gaussians.get_scaling[:, (0, 1)].prod(1),
                             gaussians.get_scaling[:, (1, 2)].prod(1)]) > 1e-2).any(0))
                        gaussians.densify_and_prune(opt.densify_grad_threshold, opt.min_opacity, scene.cameras_extent,
                                                    None, prune_min_iters=500, prune_min_T=0.1)
                        gaussians.split_ellipsoids(opt.target_radius, max_num=opt.max_num_points)
                        gaussians.double_scaling(multiplier=1.1)
                        if iteration % 5000 == 0:
                            gaussians.reset_opacity()
                            gs_bg.reset_opacity()


                elif iteration != opt.iterations:
                    if iteration % 1000 == 0:
                        gaussians.prune_points((torch.vstack(
                            [gaussians.get_scaling[:, (0, 2)].prod(1), gaussians.get_scaling[:, (0, 1)].prod(1),
                             gaussians.get_scaling[:, (1, 2)].prod(1)]) > 1e-2).any(0))
                        gaussians.densify_and_prune(opt.densify_grad_threshold, opt.min_opacity,
                                                    scene.cameras_extent, params.max_screen_size, prune_min_iters=500,
                                                    prune_min_T=0.005)
                        gaussians.split_ellipsoids(opt.target_radius, max_num=opt.max_num_points)
                        # gaussians.double_scaling(multiplier=1.1)
                        if iteration % 5000 == 0:
                            gaussians.reset_opacity()
                            gs_bg.reset_opacity()

                elif iteration == opt.iterations:
                    gaussians.prune_points((torch.vstack(
                        [gaussians.get_scaling[:, (0, 2)].prod(1), gaussians.get_scaling[:, (0, 1)].prod(1),
                         gaussians.get_scaling[:, (1, 2)].prod(1)]) > 1e-2).any(0))
                    gaussians.split_ball(opt.target_radius, max_num=opt.max_num_points)

            # Optimizer step
            if iteration <= opt.iterations:
                gs_bg.optimizer.step()
                gs_bg.optimizer.zero_grad(set_to_none=True)
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration % 1000 == 0:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gs_bg.save(), iteration, gaussians.save()),
                           model_path + "/chkpnt" + str(iteration) + ".pth")


# if __name__ == "__main__":
#     # Set up command line argument parser
#     parser = ArgumentParser(description="Training script parameters")
#     parser.add_argument('--ip', type=str, default="0.0.0.0")
#     parser.add_argument('--port', type=int, default=6009)
#     parser.add_argument("--start_checkpoint", type=str, default=None)
#     parser.add_argument('--fluid_setup', type=int, default=1)
#     args = parser.parse_args(sys.argv[1:])
#     lp = ArgumentParser(ModelParams).parse_args()
#     pp = ArgumentParser(PipelineParams).parse_args()
#     op = ArgumentParser(OptimizationParams).parse_args()
#
#     os.makedirs(args.model_path, exist_ok=True)
#     print("Optimizing " + args.model_path)
#
#     # Initialize system state (RNG)
#     safe_state(silent=False)
#
#     # Start GUI server, configure and run training
#     network_gui.init(args.ip, args.port)
#     torch.autograd.set_detect_anomaly(mode=False)
#     training(lp, op, pp, args.start_checkpoint, args.fluid_setup)
#
#     # All done
#     print("\nTraining complete.")

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
import json

import torch
from dataset import DataLoader
from model import Gaussfluids
from trans_model import TransModel
from model.gaussfluids import GaussfluidsModel
import os, sys
import yaml
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments.__init__ import ModelParams, PipelineParams, OptimizationParams
from utils.time_utils import TimeSeriesInfo
import numpy as np
from renderer.network_tools import handle_network
from renderer import network_gui
from utils.system_utils import merge_args
from dataset.readers import readFixedColmapInfo


def trans_sets(mdl: ModelParams, opt: OptimizationParams, pipe, checkpoint, fluid_setup,
               time_info: TimeSeriesInfo = None):
    yaml_conf = yaml.safe_load(open(os.path.join(source_path, 'fluid.yml')))
    merge_args(mdl, opt, yaml_conf, fluid_setup)

    with torch.no_grad():
        scene_info = readFixedColmapInfo(source_path, eval=False)
        scene = DataLoader(mdl, scene_info,is_nerf_synthetic=False)
        if opt.end_frame == -1:
            opt.end_frame = scene.time_info.num_frames - 1
        gaussians = Gaussfluids(mdl.sh_degree, base_time=scene.time_info.get_time(opt.end_frame), hidden_sizes=mdl.hidden_sizes, track_channel=mdl.track_channel)
        if checkpoint:
            (_, first_iter, model_params) = torch.load(checkpoint)
            opt_dict = gaussians.restore(model_params)
        else:
            raise Exception("No chkpnt specify")

        if time_info is None:
            time_info = scene.time_info

        bg_color = [1, 1, 1] if mdl.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        save_path = os.path.join(model_path, 'npz')
        os.makedirs(save_path, exist_ok=True)

        json.dump(time_info._asdict(), open(os.path.join(save_path, 'time_info.json'), 'w'))

        for i in range(opt.start_frame, opt.end_frame + 1):
            handle_network(pipe, gaussians, time_info, background, (i == opt.end_frame),
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


# if __name__ == "__main__":
#     # Set up command line argument parser
#     parser = ArgumentParser(description="Dump npz files script")
#     parser.add_argument('--ip', type=str, default="0.0.0.0")
#     parser.add_argument('--port', type=int, default=6009)
#     parser.add_argument("--start_checkpoint", type=str, default=None)
#     parser.add_argument('--fluid_setup', type=int, default=1)
#     args = parser.parse_args(sys.argv[1:])
#     lp = ArgumentParser(ModelParams).parse_args()
#     pp = ArgumentParser(PipelineParams).parse_args()
#     op = ArgumentParser(OptimizationParams).parse_args()
#     print("Model path: " + args.model_path)
#
#     # Initialize system state (RNG)
#     safe_state(silent=False)
#
#     # Start GUI server, configure and run training
#     network_gui.init(args.ip, args.port)
#     trans_sets(lp, op, pp, args.start_checkpoint,
#                fluid_setup=args.fluid_setup)
