import os
from dataclasses import dataclass, field

import torch
from argparse_dataclass import ArgumentParser
from torch.utils.tensorboard import SummaryWriter

from .dataset import DatasetType, detect_dataset_type
from .pipeline import scalarflow, colmap, neurofluid
from .renderer import network_gui
from .utils.general_utils import safe_state, logging_setup


@dataclass
class TrainingOptions:
    source_path: str = field(default="", metadata={"args": ["-s"], "help": "Source path", "dest": "source_path"})
    model_path: str = field(default="", metadata={"args": ["-m"], "help": "Model path", "dest": "model_path"})
    ip: str = field(default="0.0.0.0", metadata={"help": "IP address to bind the server", "dest": "ip"})
    port: int = field(default=6009, metadata={"help": "Port number for the server", "dest": "port"})
    start_checkpoint: str = field(default=None,
                                  metadata={"help": "Path to the start checkpoint", "dest": "start_checkpoint"})


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(TrainingOptions, description="Training script parameters", allow_abbrev=False)
    args, _ = parser.parse_known_args()

    # 预处理输入输出文件夹
    source_path = os.path.abspath(args.source_path)
    dataset_type: DatasetType = detect_dataset_type(source_path)
    model_path = os.path.abspath(args.model_path)
    os.makedirs(model_path, exist_ok=True)
    print("Optimizing " + model_path)

    # Initialize
    safe_state(silent=False)
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(mode=False)
    logging_setup()
    with SummaryWriter(log_dir=os.path.join(model_path, "logs")) as writer:
        if dataset_type == DatasetType.ScalarFlow:
            mp, _ = ArgumentParser(scalarflow.ModelParams, allow_abbrev=False).parse_known_args()
            pp, _ = ArgumentParser(scalarflow.PipelineParams, allow_abbrev=False).parse_known_args()
            op, _ = ArgumentParser(scalarflow.OptimizationParams, allow_abbrev=False).parse_known_args()
            scalarflow.training(source_path, model_path, mp, op, pp, args.start_checkpoint)
        elif dataset_type == DatasetType.ColmapScene:
            mp, _ = ArgumentParser(colmap.ModelParams, allow_abbrev=False).parse_known_args()
            pp, _ = ArgumentParser(colmap.PipelineParams, allow_abbrev=False).parse_known_args()
            op, _ = ArgumentParser(colmap.OptimizationParams, allow_abbrev=False).parse_known_args()
            colmap.training(source_path, model_path, mp, op, pp, args.start_checkpoint)
        elif dataset_type == DatasetType.Neurofluid:
            mp, _ = ArgumentParser(neurofluid.ModelParams, allow_abbrev=False).parse_known_args()
            pp, _ = ArgumentParser(neurofluid.PipelineParams, allow_abbrev=False).parse_known_args()
            op, _ = ArgumentParser(neurofluid.OptimizationParams, allow_abbrev=False).parse_known_args()
            neurofluid.training(source_path, model_path, mp, op, pp, writer, args.start_checkpoint)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")

    print("\nTraining complete.")
