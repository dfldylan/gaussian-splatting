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
from dataclasses import dataclass, field

from argparse_dataclass import ArgumentParser

from utils.general_utils import safe_state, logging_setup


@dataclass
class ExportOptions:
    source_path: str = field(default="", metadata={"args": ["-s"], "help": "Source path", "dest": "source_path"})
    model_path: str = field(default="", metadata={"args": ["-m"], "help": "Model path", "dest": "model_path"})
    start_checkpoint: str = field(default=None,
                                  metadata={"help": "Path to the start checkpoint", "dest": "start_checkpoint"})


from dataset import DatasetType, detect_dataset_type
from pipeline import scalarflow, colmap

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(ExportOptions, description="Dump npz files script", allow_abbrev=False)
    args, _ = parser.parse_known_args()

    # 预处理输入输出文件夹
    source_path = os.path.abspath(args.source_path)
    dataset_type: DatasetType = detect_dataset_type(source_path)
    model_path = os.path.abspath(args.model_path)
    os.makedirs(model_path, exist_ok=True)
    print("Model path: " + model_path)

    # Initialize
    safe_state(silent=False)
    logging_setup()

    if dataset_type == DatasetType.ScalarFlow:
        mp, _ = ArgumentParser(scalarflow.ModelParams, allow_abbrev=False).parse_known_args()
        pp, _ = ArgumentParser(scalarflow.PipelineParams, allow_abbrev=False).parse_known_args()
        op, _ = ArgumentParser(scalarflow.OptimizationParams, allow_abbrev=False).parse_known_args()
        scalarflow.export_npz(source_path, os.path.join(model_path, "output"), mp, op, pp, args.start_checkpoint)
    elif dataset_type == DatasetType.ColmapScene:
        mp, _ = ArgumentParser(colmap.ModelParams, allow_abbrev=False).parse_known_args()
        pp, _ = ArgumentParser(colmap.PipelineParams, allow_abbrev=False).parse_known_args()
        op, _ = ArgumentParser(colmap.OptimizationParams, allow_abbrev=False).parse_known_args()
        colmap.export_npz(source_path, os.path.join(model_path, "output"), mp, op, pp, args.start_checkpoint)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    # export_npz(lp, op, pp, args.start_checkpoint)
