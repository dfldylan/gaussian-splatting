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

from errno import EEXIST
from os import makedirs, path
import os
import uuid
from argparse import Namespace
from arguments import ModelParams, OptimizationParams

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc:  # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise


def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)


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


def merge_args(dataset: ModelParams, opt: OptimizationParams, conf):
    dataset.start_frame = conf['start_frame']
    dataset.end_frame = conf['end_frame']
    dataset.dynamics_color = [item / 256 for item in conf['rgb']]
    if 'max_radii2D' in conf.keys():
        dataset.max_radii2D = conf['max_radii2D']
    if 'bias' in conf.keys():
        dataset.color_bias = conf['bias']
    if 'eps' in conf.keys():
        dataset.eps = conf['eps']
    if 'class' in conf.keys():
        dataset.first_class = conf['class']
    if 'target_radius' in conf.keys():
        dataset.target_radius = conf['target_radius']
    if 'min_opacity' in conf.keys():
        opt.min_opacity = conf['min_opacity']
    return dataset
