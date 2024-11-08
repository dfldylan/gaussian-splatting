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
        scene = DataLoader(mdl, scene_info)
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
    parser = ArgumentParser(description="Dump npz files script")
    parser.add_argument('--ip', type=str, default="0.0.0.0")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument('--fluid_setup', type=int, default=1)
    args = parser.parse_args(sys.argv[1:])
    lp = ArgumentParser(ModelParams).parse_args()
    pp = ArgumentParser(PipelineParams).parse_args()
    op = ArgumentParser(OptimizationParams).parse_args()
    print("Model path: " + args.model_path)

    # Initialize system state (RNG)
    safe_state(silent=False)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    trans_sets(lp, op, pp, args.start_checkpoint,
               fluid_setup=args.fluid_setup)
