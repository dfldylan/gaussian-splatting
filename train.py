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
import uuid
import torch
import yaml
import numpy as np
from random import choice

from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from scene.trans_model import TransModel
from scene.cameras import Camera, MiniCam
from utils.general_utils import safe_state
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.density import compute_density
from utils.sh_utils import RGB2SH
from utils.tools import categorize, crop_main, similarity_mask
from gaussian_renderer.network_tools import handle_network

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset: ModelParams, opt: OptimizationParams, pipe, checkpoint):
    if os.path.exists(os.path.join(dataset.source_path, 'fluid.yml')):
        merge_args(dataset, yaml.safe_load(open(os.path.join(dataset.source_path, 'fluid.yml')))[args.fluid_setup])

    first_iter = 0
    _ = prepare_output_and_logger(dataset)
    scene = Scene(dataset)
    if dataset.end_frame == -1:
        dataset.end_frame = scene.time_info.num_frames - 1

    gs_bg = None
    if opt.bg_iterations > 0:
        gs_bg = GaussianModel(dataset.sh_degree)
    gaussians = GaussianModel(dataset.sh_degree)
    trans = TransModel(dataset, scene.time_info)

    if checkpoint:
        if gs_bg is not None:
            (bg_params, trans_params, first_iter, model_params) = torch.load(checkpoint)
            gs_bg.restore(bg_params, opt, position_lr_max_steps=opt.bg_iterations)
        else:
            (model_params, trans_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt, position_lr_max_steps=opt.iterations - opt.bg_iterations)
        trans.restore(trans_params, opt, reset_time=False)
    else:
        if gs_bg is not None:
            gs_bg.create_from_pcd(scene.point_cloud, scene.cameras_extent)
            gs_bg.training_setup(opt, position_lr_max_steps=opt.bg_iterations)
        gaussians.create_from_pcd(scene.point_cloud, scene.cameras_extent, init_color=dataset.dynamics_color)
        gaussians.training_setup(opt, position_lr_max_steps=opt.iterations - opt.bg_iterations)
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
                       (iteration <= int(opt.iterations)), dataset, opt.min_opacity)
        iter_start.record()

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        dynamics_iter = iteration - opt.bg_iterations
        if dynamics_iter <= 0:
            # Every 1000 its we increase the levels of SH up to a maximum degree
            gs_bg.update_learning_rate(iteration)
            if iteration % 1000 == 0:
                gs_bg.oneupSHdegree()
            viewpoint_stack = scene.getTrainCameras(frame_index=0)
            viewpoint_cam: Camera = choice(viewpoint_stack)
            gaussian_frame = gs_bg.move_0()
        else:
            if gs_bg is not None:
                # ensure only feature is require_grad on gaussian_0
                gs_bg.fixed_pose()
                gs_bg.fixed_feature_rest()
                gs_bg.fixed_feature_dc()
                gaussian_frame = gs_bg.move_0()
                gaussians.fixed_feature_rest()
            gaussians.update_learning_rate(dynamics_iter)
            if dynamics_iter > opt.warm_iterations:
                start_frame = int(
                    dataset.end_frame - min(1, 2 * dynamics_iter / (opt.iterations - opt.bg_iterations)) * (
                            dataset.end_frame - dataset.start_frame))
                frame_id = choice(range(start_frame, dataset.end_frame + 1))
            else:
                frame_id = dataset.end_frame
            viewpoint_stack = scene.getTrainCameras(frame_index=frame_id)
            viewpoint_cam: Camera = choice(viewpoint_stack)
            dt_xyz, dt_scaling, dt_rotation = trans(viewpoint_cam.time)
            gaussian_frame_dynamics = gaussians.move(dt_xyz, dt_scaling, dt_rotation)

            if gs_bg is not None:
                gaussian_frame.add_extra_gaussians(gaussian_frame_dynamics)
            else:
                gaussian_frame = gaussian_frame_dynamics

        render_pkg = render(viewpoint_cam, gaussian_frame, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
            render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        if dynamics_iter > opt.density_frome_iter:
            density = compute_density(gaussian_frame_dynamics.get_xyz, 0.002)
            density_error = 1e-16 * torch.mean(torch.square(density - torch.mean(density, dim=0, keepdim=True)))
            loss = loss + density_error
            # dt_xyz_error = 1e-10 * torch.norm(dt_xyz, dim=-1).mean() / np.abs(viewpoint_cam.time - trans.base_time)
            # loss = loss + dt_xyz_error
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
                # # Log and save
                # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                #                 testing_iterations, scene, render, trans, (pipe, background), gaussians_0, gaussians)

            if dynamics_iter > 0:
                # Densification
                if dynamics_iter <= (opt.iterations - opt.bg_iterations) / 2:
                    # Keep track of max radii in image-space for pruning
                    visibility_filter = visibility_filter[:gaussians.get_num]
                    radii = radii[:gaussians.get_num]
                    viewspace_point_tensor_grad = viewspace_point_tensor.grad[:gaussians.get_num]
                    if visibility_filter.sum().cpu().numpy() != 0:
                        gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                             radii[visibility_filter])
                        gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

                    if dynamics_iter <= opt.warm_iterations:
                        if dynamics_iter == 1000:
                            opacity_mask = (gaussians.get_opacity < opt.min_opacity).squeeze()
                            gaussians.prune_points(opacity_mask, trans=trans)
                            gaussians.prune_points(torch.tensor(
                                ~crop_main(gaussians.get_xyz.cpu().detach().numpy(), eps=dataset.eps,
                                           choose=dataset.first_class)), trans)

                        if dynamics_iter % 1000 == 0 and dynamics_iter != opt.warm_iterations:
                            size_threshold = None
                            gaussians.densify_and_prune(opt.densify_grad_threshold, opt.min_opacity,
                                                        scene.cameras_extent,
                                                        size_threshold, trans=trans)
                            if dataset.dynamics_color is not None:
                                gaussians.prune_points(~similarity_mask(gaussians._features_dc.squeeze(1),
                                                                        RGB2SH(np.asarray(dataset.dynamics_color)),
                                                                        threshold=dataset.color_bias)[0], trans)
                            if gs_bg is not None:
                                gs_bg.reset_opacity()
                            if gaussians.get_num < opt.max_num_points:
                                gaussians.split_ellipsoids(trans=trans, target_radius=dataset.target_radius)
                            gaussians.reset_opacity()

                        if dynamics_iter == opt.warm_iterations:
                            opacity_mask = (gaussians.get_opacity < opt.min_opacity).squeeze()
                            gaussians.prune_points(opacity_mask, trans=trans)
                            gaussians.prune_points(torch.tensor(
                                ~crop_main(gaussians.get_xyz.cpu().detach().numpy(), eps=dataset.eps)), trans)
                            if dataset.dynamics_color is not None:
                                color_mask = similarity_mask(gaussians._features_dc.squeeze(1),
                                                             RGB2SH(np.asarray(dataset.dynamics_color)),
                                                             threshold=dataset.color_bias)[0]
                                gaussians.prune_points(~color_mask, trans=trans)

                    if dynamics_iter > opt.warm_iterations:
                        if dynamics_iter % 1000 == 0:
                            size_threshold = None
                            gaussians.densify_and_prune(opt.densify_grad_threshold, opt.min_opacity,
                                                        scene.cameras_extent,
                                                        size_threshold, trans=trans)
                            if dataset.dynamics_color is not None:
                                gaussians.prune_points(~similarity_mask(gaussians._features_dc.squeeze(1),
                                                                        RGB2SH(np.asarray(dataset.dynamics_color)),
                                                                        threshold=dataset.color_bias)[0], trans)
                            if gs_bg is not None:
                                gs_bg.reset_opacity()
                            gaussians.split_ellipsoids(trans=trans, target_radius=dataset.target_radius,
                                                           max_num=1000000)
                            gaussians.reset_opacity()

                        if dynamics_iter % 5000 == 0:
                            gaussians.prune_points(
                                torch.tensor(~crop_main(gaussians.get_xyz.cpu().detach().numpy(), eps=dataset.eps)),
                                trans)

                        if dynamics_iter % 5000 == 2500:
                            gaussians.double_scaling()

            else:
                # Densification
                if iteration <= opt.bg_iterations / 2:
                    # Keep track of max radii in image-space for pruning
                    gs_bg.max_radii2D[visibility_filter] = torch.max(gs_bg.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])
                    gs_bg.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > 500 and iteration % 100 == 0:
                        size_threshold = None
                        gs_bg.densify_and_prune(opt.densify_grad_threshold, opt.min_opacity, scene.cameras_extent,
                                                size_threshold)

                    if iteration % 3_000 == 0 or (dataset.white_background and iteration == 500):
                        gs_bg.reset_opacity()

            # Optimizer step
            if iteration <= opt.iterations:
                if dynamics_iter >= 0:
                    gaussians.optimizer.step()
                    trans.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)
                    trans.optimizer.zero_grad(set_to_none=True)
                if gs_bg is not None:
                    gs_bg.optimizer.step()
                    gs_bg.optimizer.zero_grad(set_to_none=True)

            if iteration % 1000 == 0:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                if gs_bg is not None:
                    torch.save((gs_bg.capture(), trans.capture(), iteration, gaussians.capture()),
                               scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                else:
                    torch.save((gaussians.capture(), trans.capture(), iteration),
                               scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("../data/output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    transFunc, renderArgs, gaussians0, gaussians):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                viewpoint: Camera
                for idx, viewpoint in enumerate(config['cameras']):
                    dt_xyz, dt_scaling, dt_rotation = transFunc(viewpoint.time)
                    gausframe_extra = gaussians.move(dt_xyz, dt_scaling, dt_rotation)
                    gausframe = gaussians0.move_0()
                    gausframe.add_extra_gaussians(gausframe_extra)
                    image = torch.clamp(renderFunc(viewpoint, gausframe, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


def merge_args(dataset: ModelParams, conf):
    dataset.start_frame = conf['start_frame']
    dataset.end_frame = conf['end_frame']
    dataset.dynamics_color = [item / 256 for item in conf['rgb']]
    if 'max_radii2D' in conf.keys():
        dataset.max_radii2D = conf['max_radii2D']
    if 'bias' in conf.keys():
        dataset.color_bias = conf['bias']
    if 'eps' in conf.keys():
        dataset.eps = conf['eps']
    if 'class' in conf.keys():
        dataset.first_class = conf['class']
    if 'target_radius' in conf.keys():
        dataset.target_radius = conf['target_radius']
    return dataset


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
