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
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from utils.time_utils import TimeSeriesInfo
import numpy as np


def trans_sets(dataset: ModelParams, opt, checkpoint, time_info: TimeSeriesInfo = None):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset)
        trans = TransModel(dataset, scene.time_info, gaussians._xyz.shape[0])
        if checkpoint:
            (model_params, trans_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)
            trans.restore(trans_params)

        if time_info is None:
            time_info = scene.time_info

        save_path = os.path.join(dataset.model_path, 'npz')
        os.makedirs(save_path, exist_ok=True)

        json.dump(time_info._asdict(), open(os.path.join(save_path, 'time_info.json'), 'w'))

        for i in range(time_info.num_frames):
            time = time_info.start_time + i * time_info.time_step
            print('Frame {}, Time {}'.format(i, time))
            dt_xyz, dt_scaling, dt_rotation = trans(time)
            gaussian_frame = gaussians.move(dt_xyz, dt_scaling, dt_rotation)
            np.savez(os.path.join(save_path, '{:04}.npz'.format(i)), pos=filter_gaussian(gaussian_frame))


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
    pipeline = PipelineParams(parser)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    print("Model path: " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    trans_sets(lp.extract(args), op.extract(args), args.start_checkpoint)
