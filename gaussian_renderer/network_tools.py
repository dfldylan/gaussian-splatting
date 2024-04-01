import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from scene import GaussianModel
from gaussian_renderer import render, network_gui
from arguments import ModelParams
from scene.cameras import MiniCam
from utils.tools import similarity_mask, classify_mask
from utils.sh_utils import RGB2SH


def build_gaussframe(gaussians=None, trans=None, time=None, gaussians_bg=None):
    gaussians_bg: GaussianModel
    gaussians: GaussianModel
    if gaussians_bg is not None:
        gaussframe_0 = gaussians_bg.move_0()
    if gaussians is not None and gaussians.is_available:
        dt_xyz, dt_scaling, dt_rotation = trans(time)
        gaussframe = gaussians.move(dt_xyz, dt_scaling, dt_rotation)
    if gaussians_bg is not None:
        if gaussians is not None and gaussians.is_available:
            gaussframe_0.add_extra_gaussians(gaussframe)
        return gaussframe_0
    elif gaussians is not None and gaussians.is_available:
        return gaussframe
    else:
        return None


def handle_network(pipe, gaussians_bg, gaussians, trans, time_info, background, iter_finished, lp: ModelParams,
                   min_opacity):
    mask_manual = None
    bg_op = 0.005
    hl_op = 0.5
    if network_gui.conn == None:
        network_gui.try_connect()
    while network_gui.conn != None:
        try:
            net_image_bytes = None
            custom_cam: MiniCam
            custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer, frame, checkbox_1, checkbox_2, checkbox_3, slider_float_1, slider_float_2 = network_gui.receive()
            if custom_cam != None:
                time = time_info.get_time(frame / 100 * (lp.end_frame - lp.start_frame) + lp.start_frame)
                if checkbox_1 is False and checkbox_2 is False:
                    gaussframe = build_gaussframe(gaussians_bg=gaussians_bg, gaussians=gaussians, trans=trans,
                                                  time=time)
                elif checkbox_1 is False and checkbox_2 is True:
                    gaussframe = build_gaussframe(gaussians_bg=gaussians_bg)
                elif checkbox_1 is True and checkbox_2 is False:
                    gaussframe = build_gaussframe(gaussians=gaussians, trans=trans, time=time)
                elif checkbox_1 is True and checkbox_2 is True:
                    _gaussians: GaussianModel = copy.deepcopy(gaussians)
                    _trans = copy.deepcopy(trans)
                    _gaussians.prune_min_opacity(min_opacity, trans=_trans)
                    if mask_manual is not None:
                        opacity = np.full(_gaussians.get_opacity.shape, bg_op)  # 初始化所有点的不透明度为0.05
                        opacity[mask_manual] = hl_op
                        _gaussians.set_opacity(value=torch.tensor(opacity, dtype=torch.float, device="cuda"))

                    gaussframe = build_gaussframe(gaussians=_gaussians, trans=_trans, time=time)

                ret = render(custom_cam, gaussframe, pipe, background, scaling_modifer)
                net_image = ret["render"]
                net_image_bytes = memoryview(
                    (torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
            network_gui.send(net_image_bytes, lp.source_path)
            if do_training and (not iter_finished or not keep_alive):
                break
        except Exception as e:
            network_gui.conn = None


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
