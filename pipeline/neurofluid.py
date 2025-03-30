import logging
import os
from dataclasses import dataclass
from random import choice

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import arguments
from arguments.__init__ import ModelParams, PipelineParams
from dataset import DataLoader, DatasetInfo
from dataset.cameras import ShootModel
from dataset.readers import readNeurofluidInfo
from model.gaussfluids import Gaussfluids
from renderer import render, network_gui
from renderer.network_tools import handle_network
from utils.density import get_density_info
from utils.general_utils import safe_state, get_factor
from utils.loss_utils import l1_loss, ssim, aniso_loss, vol_loss, consistency_loss
from utils.system_utils import dump_cfg
from utils.time_utils import TimeSeriesInfo


@dataclass
class OptimizationParams(arguments.OptimizationParams):
    max_screen_size = 1000


def training(source_path, model_path, mdl: ModelParams, opt: OptimizationParams, pipe: PipelineParams,
             tb_writer: SummaryWriter, checkpoint=None):
    first_iter = 0
    dump_cfg(mdl, model_path)

    dataset, dataloader, gaussfluids = build_dataloader(source_path, mdl, opt, pipe, shuffle=True)

    if checkpoint:
        (model_params, first_iter, opt_dict) = torch.load(checkpoint)
        gaussfluids.restore(model_params)
        gaussfluids.setup(opt, dataloader.cameras_extent,
                          position_lr_max_steps=opt.iterations - opt.dynamics_iterations, opt_dict=opt_dict)
    else:
        gaussfluids.create_from_pcd(dataloader.point_cloud, init_color=pipe.dynamics_color)
        gaussfluids.setup(opt, dataloader.cameras_extent,
                          position_lr_max_steps=opt.iterations - opt.dynamics_iterations)

    bg_color = [1, 1, 1] if mdl.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(0, opt.iterations), desc="Training progress", initial=first_iter)
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        handle_network(pipe, gaussfluids, dataloader.time_info, background,
                       (iteration == int(opt.iterations)), source_path, opt.start_frame, opt.end_frame, opt.min_opacity)
        # --------  Common setup
        bg = torch.rand((3), device="cuda") if pipe.random_background else background

        # --------  Render asset preparation
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

        # --------  Render result
        render_pkg = render(shoot, gaussians, pipe, bg)
        image, viewspace_point_tensor = render_pkg["render"], render_pkg["viewspace_points"]
        visibility_filter, radii = render_pkg["visibility_filter"], render_pkg["radii"]
        T_sum, T_count = render_pkg["T_sum"], render_pkg["T_count"]

        # --------  Loss
        gt_image = shoot.image.cuda()
        l_l1 = l1_loss(image, gt_image, mask=shoot.seg_mask)
        l_bg = l1_loss(image, bg[:, None, None] * torch.ones_like(image), mask=~shoot.seg_mask)
        l_dssim = (1.0 - ssim(image, gt_image))
        tb_writer.add_scalars("Loss", global_step=iteration,
                              tag_scalar_dict={"vis/l1": l_l1.item(), "vis/bg": l_bg.item(),
                                               "vis/dssim": l_dssim.item(), })
        loss = 0.6 * l_l1 + 0.2 * l_bg + 0.2 * l_dssim
        if iteration <= opt.dynamics_iterations:
            l_sh = consistency_loss(gaussfluids.features_dc.squeeze(1))
            tb_writer.add_scalars("Loss", global_step=iteration, tag_scalar_dict={"cst/sh": l_sh.item()})
            loss = 0.9 * loss + 0.1 * l_sh
        else:
            dens, dens_mean, dens_std, dens_min, dens_max, dens_median = get_density_info(gaussians.get_xyz)
            l_density = consistency_loss(dens, target=dens_mean[None, ...], l=-2)
            l_aniso = aniso_loss(gaussfluids.get_scaling)
            l_vol = vol_loss(gaussfluids.get_scaling)
            l_opacity = consistency_loss(gaussfluids.get_opacity)
            l_sh = consistency_loss(gaussfluids.features_dc.squeeze(1), l=2)
            tb_writer.add_scalars("Loss", global_step=iteration,
                                  tag_scalar_dict={"phy/density": l_density.item(), "geo/aniso": l_aniso.item(),
                                                   "geo/vol": l_vol.item(), "cst/opacity": l_opacity.item(),
                                                   "cst/sh": l_sh.item()})
            tb_writer.add_scalars("Density", global_step=iteration,
                                  tag_scalar_dict={"mean": dens_mean.item(), "std": dens_std.item(),
                                                   "min": dens_min.item(), "max": dens_max.item(),
                                                   "median": dens_median.item()})
            factor = get_factor(min=0.9, max=0.2, left=opt.dynamics_iterations, right=opt.iterations, select=iteration)
            loss = factor * loss + (1 - factor) * (l_density + l_aniso + l_vol + l_opacity + l_sh)

        tb_writer.add_scalars("Loss", global_step=iteration, tag_scalar_dict={"loss": loss.item()})

        # --------  Backward
        loss.backward()

        # --------  Postprocessing
        with torch.no_grad():
            # --------  Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 100 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(100)
            if iteration == opt.iterations:
                progress_bar.close()

            # --------  Record grad
            viewspace_point_tensor_grad = viewspace_point_tensor.grad
            if visibility_filter.sum().cpu().numpy() != 0:
                gaussfluids.max_radii2D[visibility_filter] = torch.max(gaussfluids.max_radii2D[visibility_filter],
                                                                       radii[visibility_filter])
                gaussfluids.add_densification_stats(viewspace_point_tensor_grad, visibility_filter, T_sum.unsqueeze(-1),
                                                    T_count.unsqueeze(-1))

            # -------- Density control
            if iteration <= opt.warm_iterations:
                if iteration % 1000 == 0 and iteration != opt.warm_iterations:
                    gaussfluids.densify_and_prune(opt.densify_grad_threshold, opt.min_opacity,
                                                  dataloader.cameras_extent, opt.max_screen_size, prune_min_iters=200)
                    gaussfluids.split_ellipsoids(opt.target_radius, max_num=opt.max_num_points)
                    gaussfluids.reset_opacity()

                if iteration == opt.warm_iterations:
                    gaussfluids.prune_min_opacity(min_opacity=opt.min_opacity)

            elif iteration <= opt.dynamics_iterations:
                if iteration % 1000 == 0:
                    gaussfluids.densify_and_prune(opt.densify_grad_threshold, opt.min_opacity,
                                                  dataloader.cameras_extent, opt.max_screen_size, prune_min_iters=200)
                    gaussfluids.split_ellipsoids(opt.target_radius, max_num=opt.max_num_points)
                    gaussfluids.double_scaling(multiplier=1.1)
                    gaussfluids.reset_opacity(gaussfluids.get_opacity.mean().cpu().detach().numpy())

            elif iteration % 1000 == 0 and iteration != opt.iterations:
                gaussfluids.densify_and_prune(opt.densify_grad_threshold, opt.min_opacity,
                                              dataloader.cameras_extent, opt.max_screen_size, prune_min_iters=200)
                gaussfluids.split_ellipsoids(opt.target_radius, max_num=opt.max_num_points)

            elif iteration == opt.iterations:
                gaussfluids.prune_points((torch.vstack(
                    [gaussfluids.get_scaling[:, (0, 2)].prod(1), gaussfluids.get_scaling[:, (0, 1)].prod(1),
                     gaussfluids.get_scaling[:, (1, 2)].prod(1)]) > 1e-2).any(0))
                # gaussfluids.split_ball(opt.target_radius, max_num=opt.max_num_points)

            # --------  Optimizer step
            if iteration <= opt.iterations:
                gaussfluids.optimizer.step()
                gaussfluids.optimizer.zero_grad(set_to_none=True)

            # --------  Saving
            if iteration % 1000 == 0:
                logging.info("[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussfluids.save(), iteration, gaussfluids.optimizer.state_dict()),
                           model_path + "/chkpnt" + str(iteration) + ".pth")


def render_set(pp: PipelineParams, frame_index, shoot: ShootModel, background, render_path, gts_path, gaussfluids,
               frame_time, scaling_factor=None, opacity_factor=None, gs_color_sh: torch.Tensor = None):
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


def export_npz(source_path, output_path, mdl: ModelParams, opt: OptimizationParams, pipe: PipelineParams, checkpoint,
               time_info: TimeSeriesInfo = None, ply=True):
    with torch.no_grad():
        dataset = readNeurofluidInfo(source_path, eval=False, timestep_scaling=pipe.time_scaling)
        dataloader = DataLoader(mdl.data_device, dataset, shuffle=False, is_nerf_synthetic=True)
        if opt.end_frame == -1:
            opt.end_frame = dataloader.time_info.num_frames - 1
        gaussfluids = Gaussfluids(mdl.sh_degree, base_time=dataloader.time_info.get_time(opt.end_frame),
                                  hidden_sizes=mdl.hidden_sizes, track_channel=mdl.track_channel)
        (model_params, first_iter) = torch.load(checkpoint)
        gaussfluids.restore(model_params)

        time_info = dataloader.time_info if time_info is None else time_info

        bg_color = [1, 1, 1] if mdl.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        save_path = os.path.join(output_path, 'npz')
        os.makedirs(save_path, exist_ok=True)

        json.dump(time_info._asdict(), open(os.path.join(save_path, 'time_info.json'), 'w'))

        for i in range(opt.start_frame, opt.end_frame + 1):
            handle_network(pipe, gaussfluids, time_info, background, (i == opt.end_frame),
                           source_path, opt.start_frame, opt.end_frame, opt.min_opacity)
            time = time_info.start_time + i * time_info.time_step
            logging.info('Frame {}, Time {}'.format(i, time))
            gaussians = gaussfluids.get_static(time)
            gaussians.save_ply(os.path.join(save_path, 'ply', '{:04}.ply'.format(i))) if ply else None
            np.savez(os.path.join(save_path, '{:04}.npz'.format(i)), pos=gaussians.get_xyz.cpu().detach().numpy())


# def filter_gaussian(gaussian_frame: GaussfluidsModel):
#     xyz = gaussian_frame.get_xyz.cpu().numpy()
#     mask = gaussian_frame.get_opacity.cpu().numpy() < 0.1
#     xyz_filtered = xyz[mask[:, 0]]
#     return xyz_filtered

def build_dataloader(source_path, mdl: ModelParams, opt: OptimizationParams, pipe: PipelineParams, shuffle=True):
    dataset: DatasetInfo = readNeurofluidInfo(source_path, eval=False)
    dataloader = DataLoader(mdl.data_device, dataset, shuffle=shuffle)
    if opt.end_frame == -1:
        opt.end_frame = dataloader.time_info.num_frames - 1
    gaussfluids = Gaussfluids(mdl.sh_degree, base_time=dataloader.time_info.get_time(opt.end_frame),
                              hidden_sizes=mdl.hidden_sizes, track_channel=mdl.track_channel)
    return dataset, dataloader, gaussfluids


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
