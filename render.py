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
from scene import Scene
import os
import sys
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from scene.gaussian_model import GaussianModel
from scene.trans_model import TransModel


def render_set(pipe, frame_index, view, background, render_path, gts_path, gaussians, trans, frame_time):
    with torch.no_grad():
        dt_xyz, dt_scaling, dt_rotation = trans(frame_time)
        gaussians_frame = gaussians.move(dt_xyz, dt_scaling, dt_rotation)

        rendering = render(view, gaussians_frame, pipe, background)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:04d}'.format(frame_index) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:04d}'.format(frame_index) + ".png"))


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])

    print("Rendering " + args.model_path)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    opt = op.extract(args)
    dataset: ModelParams = lp.extract(args)
    pipe: PipelineParams = pp.extract(args)
    dataset.eval = True
    scene = Scene(dataset, shuffle=False)
    if dataset.end_frame == -1:
        dataset.end_frame = scene.time_info.num_frames - 1
    gaussians = GaussianModel(dataset.sh_degree)
    trans = TransModel(dataset, scene.time_info)
    (model_params, trans_params, first_iter) = torch.load(args.start_checkpoint)
    gaussians.restore(model_params, opt, position_lr_max_steps=opt.iterations)
    trans.restore(trans_params, opt, reset_time=False)
    view = scene.getTrainCameras()[0]

    render_path = os.path.join(dataset.model_path, "renders")
    gts_path = os.path.join(dataset.model_path, "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    for frame_index in range(scene.time_info.num_frames):
        print(frame_index)
        frame_time = scene.time_info.get_time(frame_index)
        render_set(pipe, frame_index, background=background, render_path=render_path, gts_path=gts_path,
                   gaussians=gaussians, trans=trans, view=view, frame_time=frame_time)
