import os
import time

import torch
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams


def pipeline(dataset, pipe):
    path = dataset.model_path

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    folders = os.listdir(path)
    folders = sorted(folders, key=lambda x: int(x.split('_')[-1]))
    gaussians_list = []
    for folder in folders:
        gaussians = GaussianModel(dataset.sh_degree)
        gaussians.load_ply(os.path.join(path, folder, 'point_cloud', 'iteration_30000', 'point_cloud.ply'))
        gaussians_list.append(gaussians)

    print('Load gaussians success!')
    # pipeline

    while True:
        while network_gui.conn == None:
            time.sleep(1)
            network_gui.try_connect()
        try:
            net_image_bytes = None
            (custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive,
             scaling_modifer, frame) = network_gui.receive()
            if custom_cam != None:
                frame = min(frame, len(gaussians_list) - 1)
                net_image = render(custom_cam, gaussians_list[frame], pipe, background,
                                   scaling_modifer)["render"]
                net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte()
                                             .permute(1, 2, 0).contiguous().cpu().numpy())
            network_gui.send(net_image_bytes, dataset.source_path)
        except Exception as e:
            network_gui.conn = None


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="0.0.0.0")
    parser.add_argument('--port', type=int, default=6009)
    args = parser.parse_args(sys.argv[1:])

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    pipeline(lp.extract(args), pp.extract(args))
