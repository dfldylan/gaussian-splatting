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
import json
import os
import sys
from typing import List

import numpy as np

from dataset import DatasetInfo, ShootInfo
from dataset.colmap import read_extrinsics_text, read_intrinsics_text, read_extrinsics_binary, \
    read_intrinsics_binary, read_points3D_binary, read_points3D_text, TimedImage, readColmapCameras
from dataset.tools import fetchPly, storePly, handle_time, getNerfppNorm, readCamerasFromTransforms, gen_random_points, \
    readCamerasFromScalarFlow, loadPly
from utils.sh_utils import SH2RGB
from utils.time_utils import TimeSeriesInfo


def readFixedColmapInfo(path, eval, llffhold=8, time_step=1 / 30, timestep_x=1):
    cameras_views_file = os.path.join(path, "views.txt")
    cameras_images_file = os.path.join(path, "images.txt")
    cameras_cameras_file = os.path.join(path, "cameras.txt")
    cam_views = read_extrinsics_text(cameras_views_file)
    cam_images = read_extrinsics_text(cameras_images_file)
    cam_cameras = read_intrinsics_text(cameras_cameras_file)

    # 列出路径下的所有文件和文件夹
    all_items = os.listdir(path)

    # 筛选出是文件夹且名称为纯数字的项目
    numeric_dirs = []
    for item in all_items:
        if os.path.isdir(os.path.join(path, item)):
            try:
                # 尝试将名称转换为整数
                int(item)
                numeric_dirs.append(item)
            except ValueError:
                # 如果转换失败，忽略此项
                continue

    # 将文件夹名称转换为整数并排序
    numeric_directories = numeric_dirs.sort(key=int)

    # build object dict Image
    images = {}
    image_id = 0
    for folder in numeric_directories:
        if int(folder) == 0:
            for idx, key in enumerate(cam_images):
                image_raw = cam_images[key]
                images[image_id] = TimedImage(id=image_id, qvec=image_raw.qvec, tvec=image_raw.tvec,
                                              camera_id=image_raw.camera_id, name=os.path.join(folder, image_raw.name),
                                              xys=image_raw.xys, point3D_ids=image_raw.point3D_ids, frame_id=0)
                image_id += 1
        else:
            image_raw = cam_views[int(folder)]
            for item in os.listdir(os.path.join(path, folder)):
                if os.path.splitext(item)[1][1:] != 'png':
                    continue
                images[image_id] = TimedImage(id=image_id, qvec=image_raw.qvec, tvec=image_raw.tvec,
                                              camera_id=image_raw.camera_id, name=os.path.join(folder, item),
                                              xys=image_raw.xys, point3D_ids=image_raw.point3D_ids,
                                              frame_id=int(os.path.splitext(item)[0]))
                image_id += 1

    cam_extrinsics = images
    cam_infos = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_cameras, depths_params=None,
                                  images_folder=path, depths_folder=None, seg_folder=None, time_step=time_step)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    ply_path = os.path.join(path, "points3D.ply")
    txt_path = os.path.join(path, "points3D.txt")
    if not os.path.exists(ply_path):
        xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    time_info: TimeSeriesInfo = handle_time(cam_infos)
    if timestep_x != 1:
        time_info = TimeSeriesInfo(time_info.start_time, time_info.time_step * timestep_x,
                                   time_info.num_frames // timestep_x)
    scene_info = DatasetInfo(point_cloud=pcd,
                             train_cameras=train_cam_infos,
                             test_cameras=test_cam_infos,
                             nerf_normalization=nerf_normalization,
                             time_info=time_info)
    return scene_info


def readColmapSceneInfo(path, depths, seg, images=None, time_step=1 / 30, timestep_x=1):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    ## if depth_params_file isnt there AND depths file is here -> throw error
    depths_params = None
    if depths != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)

    reading_dir = "images" if images == None else images
    cam_infos: List[ShootInfo] = readColmapCameras(cam_extrinsics=cam_extrinsics,
                                                   cam_intrinsics=cam_intrinsics,
                                                   depths_params=depths_params,
                                                   images_folder=os.path.join(path, reading_dir),
                                                   depths_folder=depths,
                                                   seg_folder=seg,
                                                   time_step=time_step,
                                                   )

    train_cam_infos = cam_infos
    test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    time_info: TimeSeriesInfo = handle_time(cam_infos)
    if timestep_x != 1:
        time_info = TimeSeriesInfo(time_info.start_time, time_info.time_step * timestep_x,
                                   time_info.num_frames // timestep_x)
    scene_info = DatasetInfo(point_cloud=pcd,
                             train_cameras=train_cam_infos,
                             test_cameras=test_cam_infos,
                             nerf_normalization=nerf_normalization,
                             time_info=time_info,
                             )
    return scene_info


def readNerfSyntheticInfo(path, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", extension)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3  # [-1.3, 1.3) for each axis
        shs = np.random.random((num_pts, 3)) / 255.0
        # pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = DatasetInfo(point_cloud=pcd,
                             train_cameras=train_cam_infos,
                             test_cameras=test_cam_infos,
                             nerf_normalization=nerf_normalization,
                             )
    return scene_info


def readNeurofluidInfo(path, eval, extension=".png", timestep_scaling=1):
    print("Reading Training Transforms")
    train_cam_infos = []
    print("Reading Test Transforms")
    test_cam_infos = []

    import joblib
    box_info = joblib.load(os.path.join(path, "box.pt"))

    for folder in os.listdir(path):
        if folder[:4] != 'view':
            continue
        sub_path = os.path.join(path, folder)
        train_cam_infos.extend(
            readCamerasFromTransforms(sub_path, "transforms_train.json", extension))
        test_cam_infos.extend(
            readCamerasFromTransforms(sub_path, "transforms_test.json", extension))

    time_info: TimeSeriesInfo = handle_time(train_cam_infos + test_cam_infos)
    if abs(timestep_scaling - 1) > 1e-5:
        time_info = TimeSeriesInfo(time_info.start_time, time_info.time_step * timestep_scaling,
                                   int((time_info.num_frames - 1) / timestep_scaling) + 1)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        ply_data = gen_random_points()
        ply_data.write(ply_path)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = DatasetInfo(point_cloud=pcd,
                             train_cameras=train_cam_infos,
                             test_cameras=test_cam_infos,
                             nerf_normalization=nerf_normalization,
                             time_info=time_info,
                             extra={'box_info': box_info})
    return scene_info


def readScalarFlowInfo(path, calib_folder, timestep_scaling=1):
    print("Reading Training Transforms")
    train_cam_infos = []
    test_cam_infos = []

    cam_infos: List[ShootInfo]
    cam_infos = readCamerasFromScalarFlow(path, calib_folder)
    train_cam_infos.extend(cam_infos)

    time_info: TimeSeriesInfo = handle_time(train_cam_infos)
    if abs(timestep_scaling - 1) > 1e-5:
        time_info = TimeSeriesInfo(time_info.start_time, time_info.time_step * timestep_scaling,
                                   int((time_info.num_frames - 1) / timestep_scaling) + 1)

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_data = gen_random_points(channel=1)
    pcd = loadPly(ply_data)
    # ply_data.write(os.path.join(path, "points3D.ply"))

    scene_info = DatasetInfo(point_cloud=pcd,
                             train_cameras=train_cam_infos,
                             test_cameras=test_cam_infos,
                             nerf_normalization=nerf_normalization,
                             time_info=time_info)
    return scene_info
