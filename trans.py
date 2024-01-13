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

import torch
from scene import Scene, GaussianModel
from scene.trans_model import TransModel
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args


def trans_sets(dataset: ModelParams, iteration: int):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        frames = scene.time_info.num_frames
        trans = TransModel(time_step=scene.time_info.time_step, load_iteration=iteration, model_path=dataset.model_path)

        save_path = os.path.join(dataset.model_path, 'frames_ply')
        makedirs(save_path, exist_ok=True)

        gaussians_list = [gaussians.to_gaussian_frame()]
        gaussians_list[-1].save_ply(os.path.join(save_path, '0.ply'))
        for i in range(1, frames):
            print('Iteration {}'.format(i))
            gaus_frame = trans(gaussians_list[-1])
            gaus_frame.save_ply(os.path.join(save_path, '{}.ply'.format(i)))
            gaussians_list.append(gaus_frame)
        return gaussians_list


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing trans script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    trans_sets(model.extract(args), args.iteration)
