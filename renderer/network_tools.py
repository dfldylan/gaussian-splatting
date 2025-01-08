import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import Gaussfluids
from model import Gaussians
from arguments import PipelineParams
from renderer import render, network_gui
from arguments.__init__ import OptimizationParams
from dataset.cameras import MiniCam
from utils.time_utils import TimeSeriesInfo
from utils.tools import similarity_mask
from utils.sh_utils import RGB2SH


def build_gaussframe(gaussians=None, time=None, gaussians_bg=None):
    gaussians_bg: Gaussfluids
    gaussians: Gaussfluids
    if gaussians_bg is not None:
        gaussframe_0 = gaussians_bg.move_0()
    if gaussians is not None and gaussians.is_available:
        gaussframe: Gaussians = gaussians.get_static(time)
    if gaussians_bg is not None:
        if gaussians is not None and gaussians.is_available:
            gaussframe_0.add_gaussians(gaussframe)
        return gaussframe_0
    elif gaussians is not None and gaussians.is_available:
        return gaussframe
    else:
        return None


def handle_network(pipe: PipelineParams, gaussfluids: Gaussfluids, time_info: TimeSeriesInfo, bg_tensor,
                   exit_flag, source_path, start_frame: int, end_frame: int, min_opacity, gaussians_bg=None):
    gaussfluids_mask_cache = None
    if network_gui.conn == None:
        network_gui.try_connect()
    while network_gui.conn != None:
        try:
            loop, exitable = network_receive_render_send(source_path, pipe, time_info, start_frame, end_frame,
                                                         gaussfluids, bg_tensor, gaussians_bg, min_opacity,
                                                         gaussfluids_mask_cache)
            if not loop and (not exit_flag or exitable):
                break
        except Exception as e:
            network_gui.conn = None


def network_receive_render_send(source_path, pipe: PipelineParams, time_info: TimeSeriesInfo, start_frame: int,
                                end_frame: int, gaussfluids: Gaussfluids, background, gaussians_bg=None,
                                min_opacity: float = 0.005, gaussfluids_mask_cache=None, bg_op=0.005, obj_op=0.5):
    net_image_bytes = None
    custom_cam: MiniCam
    custom_cam, no_loop, pipe.convert_SHs_python, pipe.compute_cov3D_python, no_exit, scaling_modifer, frame, checkbox_1, checkbox_2, checkbox_3, slider_float_1, slider_float_2 = network_gui.receive()
    loop = not no_loop
    exitable = not no_exit
    if custom_cam != None:
        time = time_info.get_time(frame / 100 * (end_frame - start_frame) + start_frame)
        gaussians = gaussfluids.get_static(time)
        if checkbox_1 is False and checkbox_2 is False:
            ret = render(custom_cam, gaussians, pipe, background, scaling_modifer)
        elif checkbox_1 is True and checkbox_2 is False:
            _background = torch.ones_like(background)
            gaussians.features_dc = torch.zeros_like(gaussians.features_dc)
            gaussians.opacity = torch.abs(gaussians.opacity)
            ret = render(custom_cam, gaussians, pipe, _background, scaling_modifer)

        elif checkbox_1 is False and checkbox_2 is True:
            gaussians = build_gaussframe(gaussians_bg=gaussians_bg)
            ret = render(custom_cam, gaussians, pipe, background, scaling_modifer)

        elif checkbox_1 is True and checkbox_2 is True:
            _gaussians: Gaussfluids = copy.deepcopy(gaussfluids)
            _gaussians.prune_min_opacity(min_opacity)
            if gaussfluids_mask_cache is not None:
                opacity = np.full(_gaussians.get_opacity.shape, bg_op)  # 初始化所有点的不透明度为0.05
                opacity[gaussfluids_mask_cache] = obj_op
                _gaussians.set_opacity(value=torch.tensor(opacity, dtype=torch.float, device="cuda"))

            gaussians = build_gaussframe(gaussians=_gaussians, time=time)
            ret = render(custom_cam, gaussians, pipe, background, scaling_modifer)

        net_image = ret["render"]
        net_image_bytes = memoryview(
            (torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
    network_gui.send(net_image_bytes, source_path)
    return loop, exitable


def print_color(labels, unique_labels, target_color=None, color_tensor=None):
    labels = labels.detach().cpu().numpy()
    unique_labels = unique_labels.detach().cpu().numpy()
    # 使用 Matplotlib 的颜色映射
    cmap = plt.get_cmap("tab10")  # 您可以选择 'viridis', 'plasma', 'inferno', 'magma', 'tab10', 'Set1' 等
    colors = [cmap(i) for i in range(10)]
    dc = np.zeros(shape=[labels.shape[0], 3])
    opacity = np.full(labels.shape, 0.05)  # 初始化所有点的不透明度为0.05
    color_map = {unique_labels[i]: colors[i] for i in range(min(10, unique_labels.shape[0]))}
    for label, color in color_map.items():
        mask = labels == label
        dc[mask] = color[:3]  # 分配颜色
        opacity[mask] = 0.5  # 分配不透明度
        if target_color is not None and torch.any(
                similarity_mask(vectors=color_tensor, target=target_color, threshold=0.65)[0]):
            opacity[mask] = 1.0
    dc = RGB2SH(dc)

    return dc, opacity
