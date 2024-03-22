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


def render_set(model_path, frame_index, views, gaussians, trans, pipeline, background):
    render_path = os.path.join(model_path, "renders", str(frame_index))
    gts_path = os.path.join(model_path, "gt", str(frame_index))

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    dt_xyz, dt_scaling, dt_rotation = trans(views[0].time)
    gaussians_frame = gaussians.move(dt_xyz, dt_scaling, dt_rotation)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians_frame, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:04d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:04d}'.format(idx) + ".png"))


def render_sets(args, model, op, pipeline, frame_index, checkpoint):
    # args = get_combined_args(args)
    dataset: ModelParams = model.extract(args)
    opt = op.extract(args)
    pipeline: PipelineParams = pipeline.extract(args)
    dataset.eval = True
    with torch.no_grad():
        scene = Scene(dataset)
        if dataset.end_frame == -1:
            dataset.end_frame = scene.time_info.num_frames - 1
        gaussians = GaussianModel(dataset.sh_degree)
        trans = TransModel(dataset, scene.time_info)
        if checkpoint:
            (model_params, trans_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt, position_lr_max_steps=opt.iterations)
            trans.restore(trans_params, opt, reset_time=False)
        else:
            raise Exception("No chkpnt specify")

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_set(dataset.model_path, frame_index, scene.getTrainCameras(frame_index=frame_index),
                   gaussians, trans, pipeline, background)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser)
    op = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])

    print("Rendering " + args.model_path)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    from scene.dataset_readers import readNeurofluidInfo

    _ = model.extract(args)
    num_frames = readNeurofluidInfo(_.source_path, _.white_background, _.eval).time_info.num_frames
    model_path = args.model_path
    for frame_index in range(num_frames):
        render_sets(args, model, op, pipeline, frame_index, args.start_checkpoint)
