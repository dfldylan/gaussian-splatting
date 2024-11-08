import logging
import os
from dataclasses import dataclass, field

import torch
import torchvision
from argparse_dataclass import ArgumentParser

import model
from arguments import PipelineParams
from dataset import DataLoader, ShootModel
from dataset import DatasetType, detect_dataset_type
# from dataset.readers import readNeurofluidInfo
from pipeline import scalarflow
# from pipeline import neurofluid
from renderer import render
from utils.general_utils import safe_state
from utils.system_utils import is_debug_mode


def render_set(pp: PipelineParams, frame_index, shoot: ShootModel, background, render_path, gts_path, gaussfluids,
               frame_time, scaling_factor=1.0):
    with torch.no_grad():
        gaussians: model.Gaussians = gaussfluids.get_static(frame_time)
        if scaling_factor != 1.0:
            new_scaling = gaussians._inverse_scaling_activation(scaling_factor * gaussians.get_scaling)
            gaussians.scaling = new_scaling
        rendering = render(shoot, gaussians, pp, background)["render"]
        gt = shoot.image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:04d}'.format(frame_index) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:04d}'.format(frame_index) + ".png"))


@dataclass
class RenderingOptions:
    source_path: str = field(metadata={"args": ["-s"], "help": "Source path", "dest": "source_path", "required": True})
    model_path: str = field(metadata={"args": ["-m"], "help": "Model path", "dest": "model_path", "required": True})
    start_checkpoint: str = field(metadata={"help": "Path to the start checkpoint", "dest": "start_checkpoint",
                                            "required": True})
    scaling_factor: float = field(metadata={"help": "Scaling factor", "dest": "scaling_factor"}, default=1.0)


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
    logging.basicConfig(level=logging.DEBUG) if is_debug_mode() else logging.basicConfig(level=logging.INFO)

    if dataset_type == DatasetType.ScalarFlow:
        mp, _ = ArgumentParser(scalarflow.ModelParams, allow_abbrev=False).parse_known_args()
        pp, _ = ArgumentParser(scalarflow.PipelineParams, allow_abbrev=False).parse_known_args()
        op, _ = ArgumentParser(scalarflow.OptimizationParams, allow_abbrev=False).parse_known_args()
        scalarflow.rendering(source_path, os.path.join(model_path, "output"), mp, op, pp, args.start_checkpoint,
                             args.scaling_factor)
    elif dataset_type == DatasetType.Neurofluid:
        pass
        # mp, _ = ArgumentParser(neurofluid.ModelParams, allow_abbrev=False).parse_known_args()
        # pp, _ = ArgumentParser(neurofluid.PipelineParams, allow_abbrev=False).parse_known_args()
        # op, _ = ArgumentParser(neurofluid.OptimizationParams, allow_abbrev=False).parse_known_args()
        # dataset = readNeurofluidInfo(source_path, mp.white_background, eval=True, timestep_x=pp.time_scaling)

        # dataloader = DataLoader(mp, dataset, shuffle=False)
        # if mp.end_frame == -1:
        #     mp.end_frame = dataloader.time_info.num_frames - 1
        # gaussfluids = model.Gaussfluids(mp.sh_degree, base_time=dataloader.time_info.get_time(op.end_frame),
        #                           hidden_sizes=mp.hidden_sizes, track_channel=mp.track_channel)
        # (model_params, first_iter) = torch.load(args.start_checkpoint)
        # opt_dict = gaussfluids.restore(model_params)
        # shoot: ShootModel = dataloader.getTrainCameras()[0]

        # render_path = os.path.join(model_path, "renders")
        # gts_path = os.path.join(model_path, "gt")
        # os.makedirs(render_path, exist_ok=True)
        # os.makedirs(gts_path, exist_ok=True)
        #
        # bg_color = [1, 1, 1] if mp.white_background else [0, 0, 0]
        # background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # for frame_index in range(dataloader.time_info.num_frames):
        #     print(frame_index)
        #     frame_time = dataloader.time_info.get_time(frame_index)
        #     render_set(pp, frame_index, background=background, render_path=render_path, gts_path=gts_path,
        #                gaussfluids=gaussfluids, shoot=shoot, frame_time=frame_time)

    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
