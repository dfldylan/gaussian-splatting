import os
import signal
import sys

import torch
from tqdm import tqdm

from pipeline.colmap import training_initialize, ModelParams, PipelineParams, OptimizationParams, build_dataloader
from renderer.network_tools import Netviewer
from utils.general_utils import get_expon_lr_func, safe_state, logging_setup
from utils.system_utils import set_output_path
from model.gaussians import Gaussians
from dataset.cameras import ShootModel
from random import choice
from renderer import render
from utils.loss_utils import l1_loss, ssim
import copy
from argparse_dataclass import ArgumentParser


def test1():
    dataset, dataloader, gaussfluids = build_dataloader(source_path, mdl, opt, pipe, shuffle=True)

    initial_path = os.path.join(save_path, 'chkpnt0.pth')
    gs_initial, gs_bg = training_initialize(opt.initial_iterations, dataloader, opt, pipe)
    torch.save((gs_initial.save(), gs_bg.save()), initial_path)

    gs_initial.save_ply(os.path.join(save_path, 'initial.ply'))
    gs_bg.save_ply(os.path.join(save_path, 'bg.ply'))

def test2(source_path, mdl: ModelParams, opt: OptimizationParams, pipe: PipelineParams,netviewer:Netviewer):
    first_iter = 0
    factor = 100
    iterations = 6400
    initial_path = os.path.join(save_path, 'chkpnt0.pth')

    dataset, dataloader, gaussfluids = build_dataloader(source_path, mdl, opt, pipe, shuffle=True)

    gaussians = Gaussians(max_sh_degree=1)
    gaussians.create_from_pcd(dataloader.point_cloud)
    gaussians.setup(opt, dataloader.cameras_extent, position_lr_max_steps=iterations)
    bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    netviewer.start_thread(pipe, gaussians, bg, source_path)

    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=iterations)

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(0, iterations), desc="Training progress", initial=first_iter)
    first_iter += 1
    for iteration in range(first_iter, iterations+1):
        bg = torch.rand((3), device="cuda")

        gaussians.update_learning_rate(iteration)
        shoot_stack = dataloader.getTrainCameras()
        shoot: ShootModel = choice(shoot_stack)

        render_pkg = render(shoot, gaussians, pipe, bg)
        image, viewspace_point_tensor = render_pkg["render"], render_pkg["viewspace_points"]
        visibility_filter, radii = render_pkg["visibility_filter"], render_pkg["radii"]
        T_sum, T_count = render_pkg["T_sum"], render_pkg["T_count"]

        gt_image = shoot.image.cuda()
        Ll1 = l1_loss(image, gt_image)
        Lssim = 1- ssim(image, gt_image)
        loss =(1- opt.lambda_dssim)* Ll1 + opt.lambda_dssim * Lssim
        if depth_l1_weight(iteration) > 0 and shoot.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = shoot.invdepthmap.cuda()
            Ll1depth_pure = torch.abs((invDepth - mono_invdepth)).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure
            loss += Ll1depth

        loss.backward()

        with torch.no_grad():
            # --------  Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % factor == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(factor)
            if iteration == opt.iterations:
                progress_bar.close()

            viewspace_point_tensor_grad = viewspace_point_tensor.grad
            if visibility_filter.sum().cpu().numpy() != 0:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter,
                                                  T_sum.unsqueeze(-1),
                                                  T_count.unsqueeze(-1))

            # -------- Density control
            if iteration % factor == 0 and iteration != iterations:
                if iteration == 4 * factor:
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.min_opacity, dataloader.cameras_extent,
                                                max_screen_size=None, prune_min_iters=factor)

                if iteration > 4 * factor and iteration % (2 * factor) == 0:
                    size_threshold = 20 if iteration > 16 * factor else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, dataloader.cameras_extent,
                                                size_threshold)

                if iteration % (16 * factor) == 0:
                    # split_aniso(gaussians)
                    gaussians.reset_opacity()

            if iteration == iterations:
                gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, dataloader.cameras_extent,
                                            size_threshold, prune_min_iters=1.8 * factor, prune_min_T=0.1)

            # --------  Optimizer step
            if iteration <= iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

    torch.save((gaussians.save()), initial_path)
    gaussians.save_ply(os.path.join(save_path, 'bg.ply'))

if __name__ == '__main__':
    source_path = '/workspace/datasets/lava'
    save_path = set_output_path()
    # Initialize
    safe_state(silent=False)

    netviewer = Netviewer()
    def cleanup(signum, frame):
        print("收到中断信号，清理资源...")
        netviewer.exit_flag = True
        netviewer.join_thread()
        netviewer.disconnect()
        sys.exit(0)
    # 注册信号
    signal.signal(signal.SIGINT, cleanup)  # Ctrl+C
    signal.signal(signal.SIGTERM, cleanup)  # kill

    torch.autograd.set_detect_anomaly(mode=False)
    logging_setup()

    mp, _ = ArgumentParser(ModelParams, allow_abbrev=False).parse_known_args()
    pp, _ = ArgumentParser(PipelineParams, allow_abbrev=False).parse_known_args()
    op, _ = ArgumentParser(OptimizationParams, allow_abbrev=False).parse_known_args()

    test2(source_path, mp, op, pp,netviewer)

    netviewer.exit_flag = True
    netviewer.join_thread()
    netviewer.disconnect()


