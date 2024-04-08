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
from scene import Scene, GaussianModel
from scene.trans_model import TransModel
from scene.gaussian_model import GaussianFrame
from tqdm import tqdm
import os, sys
import yaml
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from utils.time_utils import TimeSeriesInfo
import numpy as np
from gaussian_renderer.network_tools import handle_network
from gaussian_renderer import render, network_gui
from utils.system_utils import merge_args


def trans_sets(dataset: ModelParams, opt, pipe, checkpoint, fluid_setup, time_info: TimeSeriesInfo = None):
    yaml_conf = yaml.safe_load(open(os.path.join(dataset.source_path, 'fluid.yml')))
    merge_args(dataset, opt, yaml_conf, fluid_setup)

    with torch.no_grad():
        scene = Scene(dataset)
        if opt.end_frame == -1:
            opt.end_frame = scene.time_info.num_frames - 1
        gaussians = GaussianModel(dataset.sh_degree)
        trans = TransModel(dataset, scene.time_info, opt.end_frame)
        if checkpoint:
            (_, trans_params, first_iter, model_params) = torch.load(checkpoint)
            gaussians.restore(model_params, opt, position_lr_max_steps=opt.iterations - opt.dynamics_iterations)
            trans.restore(trans_params, opt, reset_time=False)
        else:
            raise Exception("No chkpnt specify")

        if time_info is None:
            time_info = scene.time_info

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        save_path = os.path.join(dataset.model_path, 'npz')
        os.makedirs(save_path, exist_ok=True)

        json.dump(time_info._asdict(), open(os.path.join(save_path, 'time_info.json'), 'w'))

        for i in range(opt.start_frame, opt.end_frame + 1):
            handle_network(pipe, None, gaussians, trans, time_info, background, (i == opt.end_frame),
                           dataset, opt)
            time = time_info.start_time + i * time_info.time_step
            print('Frame {}, Time {}'.format(i, time))
            dt_xyz, dt_scaling, dt_rotation = trans(time)
            gaussian_frame = gaussians.move(dt_xyz, dt_scaling, dt_rotation)
            gaussian_frame.save_ply(os.path.join(save_path, 'ply', '{:04}.ply'.format(i)))
            np.savez(os.path.join(save_path, '{:04}.npz'.format(i)), pos=gaussian_frame.get_xyz.cpu().detach().numpy())


def filter_gaussian(gaussian_frame: GaussianFrame):
    xyz = gaussian_frame.get_xyz.cpu().numpy()
    mask = gaussian_frame.get_opacity.cpu().numpy() < 0.1
    xyz_filtered = xyz[mask[:, 0]]
    return xyz_filtered


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Dump npz files script")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="0.0.0.0")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument('--fluid_setup', type=int, default=1)
    args = parser.parse_args(sys.argv[1:])
    print("Model path: " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    trans_sets(lp.extract(args), op.extract(args), pp.extract(args), args.start_checkpoint,
               fluid_setup=args.fluid_setup)