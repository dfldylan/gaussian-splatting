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
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from typing import List
from scene.cameras import Camera


class Scene:
    def __init__(self, args: ModelParams, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "views.txt")):
            scene_info = sceneLoadTypeCallbacks["FixedColmap"](args.source_path, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "box.pt")):
            scene_info = sceneLoadTypeCallbacks["Neurofluid"](args.source_path, args.white_background, args.eval,
                                                              timestep_x=args.timestep_x)
        else:
            assert False, "Could not recognize scene type!"

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale,
                                                                            args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale,
                                                                           args)

        self.time_info = scene_info.time_info
        self.extra = scene_info.extra
        self.point_cloud = scene_info.point_cloud

    # def save(self, iteration):
    #     point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
    #     self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0, frame_index=None) -> List[Camera]:
        cameraList = self.train_cameras[scale]
        if frame_index is not None:
            if frame_index >= self.time_info.num_frames:
                return []
            find_time = self.time_info.start_time + frame_index * self.time_info.time_step
            return [camera for camera in cameraList if abs(camera.time - find_time) < 0.001]
        else:
            return cameraList

    def getTestCameras(self, scale=1.0, frame_index=None) -> List[Camera]:
        cameraList = self.test_cameras[scale]
        if frame_index is not None:
            if frame_index >= self.time_info.num_frames:
                return []
            find_time = self.time_info.start_time + frame_index * self.time_info.time_step
            return [camera for camera in cameraList if abs(camera.time - find_time) < 0.001]
        else:
            return cameraList
