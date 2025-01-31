import os
from dataclasses import dataclass, field

from argparse_dataclass import ArgumentParser

from dataset import DatasetType, detect_dataset_type
from pipeline import scalarflow, colmap
from utils.general_utils import safe_state, logging_setup


@dataclass
class RenderingOptions:
    source_path: str = field(metadata={"args": ["-s"], "help": "Source path", "dest": "source_path", "required": True})
    model_path: str = field(metadata={"args": ["-m"], "help": "Model path", "dest": "model_path", "required": True})
    start_checkpoint: str = field(metadata={"help": "Path to the start checkpoint", "dest": "start_checkpoint",
                                            "required": True})
    scaling_factor: str = field(metadata={"help": "Scaling factor", "dest": "scaling_factor"}, default=None)
    opacity_factor: str = field(metadata={"help": "Opacity factor", "dest": "opacity_factor"}, default=None)
    bg_color: str = field(metadata={"help": "Background color", "dest": "bg_color"}, default=None)
    gs_color: str = field(metadata={"help": "Gaussis color", "dest": "gs_color"}, default=None)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(RenderingOptions, description="Rendering script parameters", allow_abbrev=False)
    args, _ = parser.parse_known_args()

    # 预处理输入输出文件夹
    source_path = os.path.abspath(args.source_path)
    dataset_type: DatasetType = detect_dataset_type(source_path)
    model_path = os.path.abspath(args.model_path)
    print("Rendering " + args.model_path)
    # Initialize
    safe_state(silent=False)
    logging_setup()

    if dataset_type == DatasetType.ScalarFlow:
        mp, _ = ArgumentParser(scalarflow.ModelParams, allow_abbrev=False).parse_known_args()
        pp, _ = ArgumentParser(scalarflow.PipelineParams, allow_abbrev=False).parse_known_args()
        op, _ = ArgumentParser(scalarflow.OptimizationParams, allow_abbrev=False).parse_known_args()
        scalarflow.rendering(source_path, os.path.join(model_path, "output"), mp, op, pp, args.start_checkpoint,
                             args.scaling_factor, args.opacity_factor, args.bg_color, args.gs_color)
    elif dataset_type == DatasetType.Neurofluid:
        pass
    elif dataset_type == DatasetType.ColmapScene:
        mp, _ = ArgumentParser(colmap.ModelParams, allow_abbrev=False).parse_known_args()
        pp, _ = ArgumentParser(colmap.PipelineParams, allow_abbrev=False).parse_known_args()
        op, _ = ArgumentParser(colmap.OptimizationParams, allow_abbrev=False).parse_known_args()
        colmap.rendering(source_path, os.path.join(model_path, "output"), mp, op, pp, args.start_checkpoint,
                         args.scaling_factor, args.opacity_factor, args.bg_color, args.gs_color)


    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
