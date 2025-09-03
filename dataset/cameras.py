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

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn

from . import ShootInfo
from ..utils.general_utils import PILtoTorch
from ..utils.graphics_utils import getWorld2View2, getProjectionMatrix


class ShootModel(nn.Module):
    def __init__(self, shoot_info: ShootInfo, zfar=100.0, znear=0.01, trans=np.array([0.0, 0.0, 0.0]), scale=1.0,
                 is_nerf_synthetic=False):
        super(ShootModel, self).__init__()
        self.shoot_info: ShootInfo = shoot_info
        self.FoVx = self.shoot_info.FovX
        self.FoVy = self.shoot_info.FovY
        self.time = self.shoot_info.time

        self.is_nerf_synthetic = is_nerf_synthetic
        self.invdepthmap = None
        self.depth_reliable = False
        self.depth_mask = None

        self.world_view_transform = (torch.tensor(getWorld2View2(self.shoot_info.R, self.shoot_info.T, trans, scale))
                                     .transpose(0, 1).cuda())
        self.projection_matrix = (getProjectionMatrix(znear=znear, zfar=zfar, fovX=self.FoVx, fovY=self.FoVy)
                                  .transpose(0, 1).cuda())
        self.full_proj_transform = ((self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0)))
                                    .squeeze(0))
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        self._image = None
        self._image_width = None
        self._image_height = None

    def _load_image(self):
        if self._image is None:
            origin_image = Image.open(self.shoot_info.image_path)
            orig_w, orig_h = origin_image.size

            # depth
            if self.shoot_info.depth_path != "":
                depth_path = self.shoot_info.depth_path
                try:
                    if self.is_nerf_synthetic:
                        invdepthmap = cv2.imread(depth_path, -1).astype(np.float32) / 512
                    else:
                        invdepthmap = cv2.imread(depth_path, -1).astype(np.float32) / float(2 ** 16)

                except FileNotFoundError:
                    print(f"Error: The depth file at path '{depth_path}' was not found.")
                    raise
                except IOError:
                    print(
                        f"Error: Unable to open the image file '{depth_path}'. It may be corrupted or an unsupported format.")
                    raise
                except Exception as e:
                    print(
                        f"An unexpected error occurred when trying to read depth at {depth_path}: {e}")
                    raise
            else:
                invdepthmap = None

            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                          "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1

            scale = float(global_down)
            resolution = (int(orig_w / scale), int(orig_h / scale))

            resized_image: torch.Tensor = PILtoTorch(origin_image, resolution)  # (channel, height, width)

            loaded_mask = None
            if resized_image.shape[0] == 4:
                loaded_mask = resized_image[3:4, ...]
                gt_image = resized_image[:3, ...]
            elif resized_image.shape[0] == 3:
                gt_image = resized_image[:3, ...]
            elif resized_image.shape[0] == 1:
                gt_image = resized_image[:1, ...]
            else:
                raise ValueError(f"Unexpected image shape: {resized_image.shape}")

            # seg
            if self.shoot_info.seg_path != "":
                seg_path = self.shoot_info.seg_path
                # 读取二值图像
                binary_mask = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)

                # 确认图像是否成功加载
                if binary_mask is None:
                    raise ValueError("Failed to load the binary mask image.")

                # # 确认图像是否只有两种像素值（0和255）
                # unique_values = np.unique(binary_mask)
                # print("Unique pixel values in the image:", unique_values)

                # 如果需要将其转换为布尔掩码 (True/False)
                self.seg_mask = torch.from_numpy(cv2.resize(binary_mask, resolution) > 0).cuda()
            elif loaded_mask is not None:
                # 如果加载了透明度掩码，则将其转换为布尔掩码 (True/False)
                self.seg_mask = (loaded_mask[0, ...] > 0).cuda()

            self._image = gt_image.clamp(0.0, 1.0)
            self._image_width = self._image.shape[2]
            self._image_height = self._image.shape[1]

            if loaded_mask is not None:
                self._image *= loaded_mask
            else:
                self._image *= torch.ones((1, self._image_height, self._image_width))

            if invdepthmap is not None:
                self.depth_mask = torch.ones_like(resized_image[0:1, ...].cuda())
                self.invdepthmap = cv2.resize(invdepthmap, resolution)
                self.invdepthmap[self.invdepthmap < 0] = 0
                self.depth_reliable = True

                if self.shoot_info.depth_params is not None:
                    depth_params = self.shoot_info.depth_params
                    if depth_params["scale"] < 0.2 * depth_params["med_scale"] or depth_params["scale"] > 5 * \
                            depth_params["med_scale"]:
                        self.depth_reliable = False
                        self.depth_mask *= 0

                    if depth_params["scale"] > 0:
                        self.invdepthmap = self.invdepthmap * depth_params["scale"] + depth_params["offset"]

                if self.invdepthmap.ndim != 2:
                    self.invdepthmap = self.invdepthmap[..., 0]
                self.invdepthmap = torch.from_numpy(self.invdepthmap[None]).cuda()

    @property
    def image(self):
        self._load_image()
        return self._image

    @property
    def image_width(self):
        self._load_image()
        return self._image_width

    @property
    def image_height(self):
        self._load_image()
        return self._image_height


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]


WARNED = False


def cameraList_from_camInfos(cam_infos, is_nerf_synthetic):
    return [ShootModel(shoot_info=cam_info, is_nerf_synthetic=is_nerf_synthetic) for cam_info in cam_infos]
