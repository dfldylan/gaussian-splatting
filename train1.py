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
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from scene.trans_model import TransModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import time

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, checkpoint, trans_checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    opt.position_lr_max_steps = opt.iterations
    trans = TransModel(time_step=scene.time_info.time_step, model_path=dataset.model_path, box_info=scene.extra['box_info'])
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        print("restore guas_params from {}".format(checkpoint))
        gaussians.restore(model_params, opt)
    if trans_checkpoint:
        # trans_params = torch.load(trans_checkpoint)
        print("restore trans_params from {}".format(trans_checkpoint))
        # trans.model.load_state_dict(trans_params, strict=False)
        import torch.nn.init as init

        model = trans.model  # 您的模型实例
        checkpoint = torch.load(trans_checkpoint)  # 加载预训练模型

        for name, param in model.named_parameters():
            if name in checkpoint:
                if param.size() == checkpoint[name].size():
                    # 尺寸匹配，直接加载参数
                    param.data.copy_(checkpoint[name])
                else:
                    # 尺寸不匹配，重新初始化
                    if len(param.shape) == 1:  # 偏置项通常是一维的
                        init.zeros_(param)  # 偏置项可以初始化为0
                    else:
                        init.xavier_uniform_(param)  # 例如使用Xavier均匀分布初始化权重

    os.makedirs(os.path.join(scene.model_path, 'ckpt_gaus'), exist_ok=True)
    os.makedirs(os.path.join(scene.model_path, 'ckpt_trans'), exist_ok=True)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    def get_loss(image, gt_image):
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        return Ll1, loss

    first_iter = first_iter + 1
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="TRAIN")
    last_save_time = time.time()
    save_interval = 600
    ema_loss_for_log = 0.0

    # pipeline
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                (custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive,
                 scaling_modifer, frame, _) = network_gui.receive()  # frame from [0,100)
                if custom_cam != None:
                    frame = min(frame, len(gaussians_list) - 1)
                    net_image = render(custom_cam, gaussians_list[frame], pipe, background,
                                       scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte()
                                                 .permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if (do_training and ((iteration < int(opt.iterations)) or not keep_alive)):
                    break
            except Exception as e:
                network_gui.conn = None

        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        gaussians_list = []
        iter_start.record()
        for i in range(0, scene.time_info.num_frames):

            if i == 0:
                gaussians_i = gaussians.to_gaussian_frame()
            else:
                gaussians_i = trans(gaussians_i.clone_detached())
            gaussians_list.append(gaussians_i)
            viewpoint_stack = scene.getTrainCameras(i)
            Ll1_list = []
            loss_list = []
            for viewpoint_cam in viewpoint_stack:
                gt = viewpoint_cam.original_image.cuda()
                bg = torch.rand((3), device="cuda") if opt.random_background else background
                render_pkg = render(viewpoint_cam, gaussians_i, pipe, bg)
                image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg[
                    "viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                Ll1, loss = get_loss(image, gt)
                Ll1_list.append(Ll1)
                loss_list.append(loss)
            Ll1 = torch.mean(torch.stack(Ll1_list))
            loss = torch.mean(torch.stack(loss_list))
            loss.backward()
            with torch.no_grad():
                # Optimizer step
                gaussians.optimizer.step()
                trans.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                trans.optimizer.zero_grad(set_to_none=True)

                # Densification
                if iteration < opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                         radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                # # 检查是否到达保存时间
                # current_time = time.time()
                # if current_time - last_save_time >= save_interval:
                #     # 保存模型
                #     print("[ITER {}] Saving Checkpoint".format(i))
                #     torch.save((gaussians.capture(), i), os.path.join(scene.model_path, 'ckpt_gaus', "{}_{}.pth".
                #                                                       format(i, time.strftime("%m%d%H%M%S",
                #                                                                               time.localtime(
                #                                                                                   current_time)))))
                #     torch.save(trans.model.state_dict(), os.path.join(scene.model_path, 'ckpt_trans', "{}_{}.pth".
                #                                                       format(i, time.strftime("%m%d%H%M%S",
                #                                                                               time.localtime(
                #                                                                                   current_time)))))
                #     # 更新最后保存时间
                #     last_save_time = current_time

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            if iteration % 1 == 0:
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(1)
            if iteration == opt.iterations:
                progress_bar.close()

            # Densification
            if iteration < opt.densify_until_iter:

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent,
                                                size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Log and save
            if iteration % 10 == 0:
                training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                                scene, render, (pipe, background))
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration),
                           os.path.join(scene.model_path, 'ckpt_gaus', "{}.pth".format(iteration)))
                torch.save(trans.model.state_dict(),
                           os.path.join(scene.model_path, 'ckpt_trans', "{}.pth".format(iteration)))


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


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, scene: Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--gaus_checkpoint", type=str, default=None)
    parser.add_argument("--trans_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.gaus_checkpoint, args.trans_checkpoint,
             args.debug_from)

    # All done
    print("\nTraining complete.")
