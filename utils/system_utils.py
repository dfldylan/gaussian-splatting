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

import os
from argparse import Namespace
from errno import EEXIST
from os import makedirs, path

from arguments.__init__ import ModelParams, OptimizationParams
import time

OUTPUT_PATH = "/tmp/gaussfluids"

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


def dump_cfg(args, folder):
    with open(os.path.join(folder, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))


def merge_args(dataset: ModelParams, opt: OptimizationParams, conf, fluid_setup):
    opt.start_frame = conf[fluid_setup]['start_frame']
    opt.end_frame = conf[fluid_setup]['end_frame']

    conf = conf[0]
    opt.static_start = conf['start_frame']
    opt.static_end = conf['end_frame']

    pipe.dynamics_color = [item / 256 for item in conf['rgb']]
    opt.target_radius = conf['target_radius']
    opt.min_opacity = conf['min_opacity']
    opt.eps = conf['eps']

    return dataset


import sys


def is_debug_mode():
    return sys.gettrace() is not None

def set_output_path():
    output_path = os.path.join(OUTPUT_PATH, time.strftime("%Y%m%d%H%M%S",time.localtime()))
    os.makedirs(output_path, exist_ok=True)
    return output_path
