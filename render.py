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
from mdl import DataLoader, ShootModel
import os
import sys
from os import makedirs
from renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse_dataclass import ArgumentParser
from arguments.__init__ import ModelParams, PipelineParams, OptimizationParams
from model import Gaussfluids
from trans_model import TransModel
from dataset_readers import readNeurofluidInfo


def render_set(pipe, frame_index, view: ShootModel, background, render_path, gts_path, gaussians, frame_time):
    with torch.no_grad():
        gaussians_frame = gaussians.get_static(frame_time)

        rendering = render(view, gaussians_frame, pipe, background)["render"]
        gt = view.image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:04d}'.format(frame_index) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:04d}'.format(frame_index) + ".png"))


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    lp = ArgumentParser(ModelParams).parse_args()
    pp = ArgumentParser(PipelineParams).parse_args()
    op = ArgumentParser(OptimizationParams).parse_args()

    print("Rendering " + args.model_path)
    # Initialize system state (RNG)
    safe_state(silent=False)

    opt = op
    mdl: ModelParams = lp
    pipe: PipelineParams = pp
    scene_info = readNeurofluidInfo(source_path, mdl.white_background, eval=True, timestep_x=pipe.time_scaling)

    scene = DataLoader(mdl, scene_info, shuffle=False)
    if mdl.end_frame == -1:
        mdl.end_frame = scene.time_info.num_frames - 1
    gaussians = Gaussfluids(mdl.sh_degree, base_time=scene.time_info.get_time(opt.end_frame), hidden_sizes=mdl.hidden_sizes,track_channel=mdl.track_channel)
    (model_params, first_iter) = torch.load(args.start_checkpoint)
    opt_dict = gaussians.restore(model_params)
    view: ShootModel = scene.getTrainCameras()[0]

    render_path = os.path.join(model_path, "renders")
    gts_path = os.path.join(model_path, "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    bg_color = [1, 1, 1] if mdl.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    for frame_index in range(scene.time_info.num_frames):
        print(frame_index)
        frame_time = scene.time_info.get_time(frame_index)
        render_set(pipe, frame_index, background=background, render_path=render_path, gts_path=gts_path,
                   gaussians=gaussians, view=view, frame_time=frame_time)
