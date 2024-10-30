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
import random
from enum import IntEnum
from typing import List, NamedTuple

import numpy as np
import torch


class ShootInfo(NamedTuple):
    uid: int
    R: np.array  # R's transpose of w2c matrix, same to R of c2w matrix
    T: np.array  # T of w2c matrix
    FovY: np.array
    FovX: np.array
    depth_params: dict
    image_path: str
    image_name: str
    depth_path: str
    seg_path: str
    width: int
    height: int
    time: float = 0.0


class DatasetType(IntEnum):
    NerfSynthetic = 0
    ColmapScene = 1
    Neurofluid = 2
    FixedColmap = 3
    ScalarFlow = 4


def detect_dataset_type(path):
    # 检查不同的数据集类型
    if os.path.exists(os.path.join(path, "input", "cam")):
        return DatasetType.ScalarFlow
    elif os.path.exists(os.path.join(path, "sparse")):
        return DatasetType.ColmapScene
    elif os.path.exists(os.path.join(path, "views.txt")):
        return DatasetType.FixedColmap
    elif os.path.exists(os.path.join(path, "transforms_train.json")):
        print("Found transforms_train.json file, assuming Blender data set!")
        return DatasetType.NerfSynthetic
    elif os.path.exists(os.path.join(path, "box.pt")):
        return DatasetType.Neurofluid
    else:
        raise ValueError("Could not recognize scene type!")


from utils.graphics_utils import BasicPointCloud
from utils.time_utils import TimeSeriesInfo


class DatasetInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    time_info: TimeSeriesInfo = TimeSeriesInfo(0, 0, 1)
    extra: dict = {}


from dataset.cameras import ShootModel, cameraList_from_camInfos


class DataLoader:
    def __init__(self, data_device, scene_info: DatasetInfo, shuffle=True, is_nerf_synthetic=True):
        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")

        print("Loading Training Cameras")
        self.train_cameras = cameraList_from_camInfos(scene_info.train_cameras, is_nerf_synthetic=is_nerf_synthetic)
        print("Loading Test Cameras")
        self.test_cameras = cameraList_from_camInfos(scene_info.test_cameras, is_nerf_synthetic=is_nerf_synthetic)

        self.time_info = scene_info.time_info
        self.extra = scene_info.extra
        self.point_cloud = scene_info.point_cloud

    def getTrainCameras(self, frame_index=None) -> List[ShootModel]:
        if frame_index is not None:
            if frame_index >= self.time_info.num_frames:
                return []
            find_time = self.time_info.start_time + frame_index * self.time_info.time_step
            return [camera for camera in self.train_cameras if abs(camera.time - find_time) < 0.001]
        else:
            return self.train_cameras
