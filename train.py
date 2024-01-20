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
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from scene.trans_model import TransModel
from scene.cameras import Camera, MiniCam
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.density import compute_density

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def build_gausframe(gaussians0, gaussians, trans, time):
    gaussians0: GaussianModel
    gaussians: GaussianModel
    gausframe = gaussians0.move_0()
    if gaussians.is_available:
        dt_xyz, dt_scaling, dt_rotation = trans(time)
        gausframe.add_static_gaussians(gaussians.move(dt_xyz, dt_scaling, dt_rotation))
    return gausframe


def handle_network(pipe, gaussians0, gaussians, trans, time_info, background, iter_finished, msg):
    if network_gui.conn == None:
        network_gui.try_connect()
    while network_gui.conn != None:
        try:
            net_image_bytes = None
            custom_cam: MiniCam
            custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer, frame = network_gui.receive()
            if custom_cam != None:
                gausframe = build_gausframe(gaussians0, gaussians, trans,
                                            time_info.get_time(frame / 100 * time_info.num_frames))
                net_image = render(custom_cam, gausframe, pipe, background, scaling_modifer)["render"]
                net_image_bytes = memoryview(
                    (torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
            network_gui.send(net_image_bytes, msg)
            if do_training and (iter_finished or not keep_alive):
                break
        except Exception as e:
            network_gui.conn = None


from scene.dataset_readers import gen_random_points, fetchPly


def training(dataset, opt, pipe, testing_iterations, checkpoint_iterations, checkpoint, debug_from):
    opt: OptimizationParams
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    scene = Scene(dataset)
    gaussians0 = GaussianModel(dataset.sh_degree)
    gaussians = GaussianModel(dataset.sh_degree)
    trans = TransModel(dataset, scene.time_info)
    if checkpoint:
        try:
            (model0_params, trans_params, first_iter, model_params) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)
        except:
            (model0_params, trans_params, first_iter) = torch.load(checkpoint)
        gaussians0.restore(model0_params, opt)
        # trans.restore(trans_params)
        # gen_random_points('temp.ply', num_pts=10000)
        gaussians.create_from_pcd(scene.point_cloud, scene.cameras_extent)  # todo delete
        trans.set_model(dataset, gaussians.get_num)
    else:
        gaussians0.create_from_pcd(scene.point_cloud, scene.cameras_extent)
        gen_random_points('temp.ply')
        gaussians.create_from_pcd(fetchPly('temp.ply'), scene.cameras_extent)
        trans.set_model(dataset, gaussians.get_num)
    gaussians0.training_setup(opt)
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(0, opt.iterations), desc="Training progress", initial=first_iter)
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        handle_network(pipe, gaussians0, gaussians, trans, scene.time_info, background,
                       (iteration < int(opt.iterations)), dataset.source_path)
        iter_start.record()

        gaussians0.update_learning_rate(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % opt.up_SHdegree_interval == 0:
            gaussians0.oneupSHdegree()

        fake_iter = iteration - opt.static_until_iter
        if fake_iter > 0:
            gaussians.update_learning_rate(fake_iter)
            if iteration % opt.up_SHdegree_interval == 0:
                gaussians.oneupSHdegree()

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        if fake_iter > 0:
            viewpoint_stack = scene.getTrainCameras(frame_index=scene.time_info.num_frames-1)
            viewpoint_cam: Camera = choice(viewpoint_stack)
            dt_xyz, dt_scaling, dt_rotation = trans(viewpoint_cam.time)
            gaussian_frame = gaussians.move(dt_xyz, dt_scaling, dt_rotation)
            gaussian_frame_0 = gaussians0.move_0()
            gaussian_frame.add_static_gaussians(gaussian_frame_0)
        else:
            viewpoint_stack = scene.getTrainCameras(frame_index=0)
            viewpoint_cam: Camera = choice(viewpoint_stack)
            gaussian_frame = gaussians0.move_0()
        render_pkg = render(viewpoint_cam, gaussian_frame, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
            render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        # if iteration > opt.density_loss_from_iter:
        #     density = compute_density(gaussian_frame.get_xyz, 0.1)
        #     density_error = 1e-10 * torch.mean(torch.square(density - torch.mean(density, dim=0, keepdim=True)))
        #     loss = loss + density_error
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10_0 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10_0)
            if iteration == opt.iterations:
                progress_bar.close()
                # Log and save
                training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                                testing_iterations, scene, render, trans, (pipe, background), gaussians0, gaussians)
            # if (iteration in saving_iterations):
            #     print("\n[ITER {}] Saving Gaussians".format(iteration))
            #     scene.save(iteration)

            if fake_iter > 0:
                # Densification
                if fake_iter < opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    visibility_filter = visibility_filter[:gaussians.get_num]
                    radii = radii[:gaussians.get_num]
                    if visibility_filter.sum().cpu().numpy() == 0:
                        raise Exception('visibility_filter is all False @ iter {}'.format(iteration))
                    # viewspace_point_tensor = viewspace_point_tensor[:gaussians.get_num].clone()
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                         radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter,
                                                      slice=range(gaussians.get_num))

                    if fake_iter > opt.densify_from_iter and fake_iter % opt.densification_interval == 0:
                        size_threshold = 20 if fake_iter > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent,
                                                    size_threshold, trans=trans)

                    if fake_iter % opt.opacity_reset_interval == 0 or (
                            dataset.white_background and fake_iter == opt.densify_from_iter):
                        gaussians.reset_opacity()
            else:
                # Densification
                if iteration < opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussians0.max_radii2D[visibility_filter] = torch.max(gaussians0.max_radii2D[visibility_filter],
                                                                          radii[visibility_filter])
                    gaussians0.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians0.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent,
                                                     size_threshold)

                    if iteration % opt.opacity_reset_interval == 0 or (
                            dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians0.reset_opacity()

            # Optimizer step
            if iteration <= opt.iterations:
                if fake_iter > 0:
                    gaussians.optimizer.step()
                    trans.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)
                    trans.optimizer.zero_grad(set_to_none=True)
                else:
                    gaussians0.optimizer.step()
                    gaussians0.optimizer.zero_grad(set_to_none=True)
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                if fake_iter > 0:
                    torch.save((gaussians0.capture(), trans.capture(), iteration, gaussians.capture()),
                               scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                else:
                    torch.save((gaussians0.capture(), trans.capture(), iteration),
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
                    gausframe = gaussians.move(dt_xyz, dt_scaling, dt_rotation)
                    gausframe_0 = gaussians0.move_0()
                    gausframe.add_static_gaussians(gausframe_0)
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


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="0.0.0.0")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=list(range(0, 60_000_0, 500_0)))
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=list(range(0, 60_000_0, 100_0)))
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.checkpoint_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.checkpoint_iterations,
             args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
