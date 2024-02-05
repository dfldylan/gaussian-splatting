import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from scene import GaussianModel
from gaussian_renderer import render, network_gui
from arguments import ModelParams
from scene.cameras import MiniCam
from utils.tools import categorize, similarity_mask
from utils.sh_utils import RGB2SH


def build_gausframe(gaussians=None, trans=None, time=None, gaussians_bg=None):
    gaussians_bg: GaussianModel
    gaussians: GaussianModel
    if gaussians_bg is not None:
        gausframe_0 = gaussians_bg.move_0()
    if gaussians is not None and gaussians.is_available:
        dt_xyz, dt_scaling, dt_rotation = trans(time)
        gausframe = gaussians.move(dt_xyz, dt_scaling, dt_rotation)
    if gaussians_bg is not None:
        if gaussians is not None and gaussians.is_available:
            gausframe_0.add_extra_gaussians(gausframe)
        return gausframe_0
    elif gaussians is not None and gaussians.is_available:
        return gausframe
    else:
        return None


def handle_network(pipe, gaussians_bg, gaussians, trans, time_info, background, iter_finished, lp: ModelParams,
                   min_opacity):
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
                    gausframe = build_gausframe(gaussians_bg=gaussians_bg, gaussians=gaussians, trans=trans, time=time)
                elif checkbox_1 is False and checkbox_2 is True:
                    gausframe = build_gausframe(gaussians_bg=gaussians_bg)
                elif checkbox_1 is True:
                    _gaussians = copy.deepcopy(gaussians)
                    _trans = copy.deepcopy(trans)
                    if checkbox_2 is True:
                        mask = \
                            similarity_mask(_gaussians._features_dc.squeeze(1), RGB2SH(np.asarray(lp.dynamics_color)),
                                            threshold=slider_float_1)[0]
                        _gaussians.prune_points(~mask, trans=_trans)

                    if checkbox_3 is True:
                        prune_mask = (_gaussians.get_opacity < min_opacity).squeeze()
                        _gaussians.prune_points(prune_mask, trans=_trans)
                        points = _gaussians.get_xyz.cpu().detach().numpy()
                        unique_labels, counts, labels = categorize(points, eps=slider_float_2 / 10)
                        # 使用 Matplotlib 的颜色映射
                        cmap = plt.get_cmap("tab10")  # 您可以选择 'viridis', 'plasma', 'inferno', 'magma', 'tab10', 'Set1' 等
                        colors = [cmap(i) for i in range(10)]
                        dc = np.zeros_like(points)
                        opacity = np.full(labels.shape, 0.05)  # 初始化所有点的不透明度为0.05
                        # select_crop = None if abs(slider_float_2 - 1) < 1e-5 else int(10 * slider_float_2)
                        # if select_crop is None:
                        #     color_map = {unique_labels[i]: colors[i] for i in range(10)}
                        # else:
                        #     color_map = {unique_labels[select_crop]: colors[select_crop]}
                        color_map = {unique_labels[i]: colors[i] for i in range(min(10, unique_labels.shape[0]))}
                        for label, color in color_map.items():
                            mask = labels == label
                            dc[mask] = color[:3]  # 分配颜色
                            opacity[mask] = 0.8  # 分配不透明度
                        dc = RGB2SH(dc)

                        _gaussians.set_opacity(value=torch.tensor(opacity, dtype=torch.float, device="cuda"))
                        _gaussians.set_featrue_dc(
                            new_dc=torch.tensor(dc, dtype=torch.float, device="cuda").unsqueeze(1))

                    gausframe = build_gausframe(gaussians=_gaussians, trans=_trans, time=time)

                net_image = render(custom_cam, gausframe, pipe, background, scaling_modifer)["render"]
                net_image_bytes = memoryview(
                    (torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
            network_gui.send(net_image_bytes, lp.source_path)
            if do_training and (iter_finished or not keep_alive):
                break
        except Exception as e:
            network_gui.conn = None
