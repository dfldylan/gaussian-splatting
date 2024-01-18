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
import copy
import os
import sys
from PIL import Image
from typing import NamedTuple, List, Dict
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text, TimedImage
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from utils.time_utils import TimeSeriesInfo


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    time: float


def handle_time(cam_infos: List[CameraInfo]) -> TimeSeriesInfo:
    if not cam_infos:
        raise ValueError("cam_infos is empty")

    # 提取所有时间信息并去重
    times = sorted(set(cam_info.time for cam_info in cam_infos))

    # 计算相邻时刻的差值
    time_differences = [t2 - t1 for t1, t2 in zip(times, times[1:])]

    # 找到最小的非零时间差异作为时间步长的估计
    min_non_zero_diff = min(diff for diff in time_differences if diff > 0)

    # 检查所有时间差异是否是估计步长的整数倍
    if not all(np.isclose(diff % min_non_zero_diff, 0) for diff in time_differences):
        raise ValueError("Time intervals are not multiples of a single minimum step")

    # 确定开始时间和帧数
    start_time = times[0]
    # 计算总时间跨度
    total_time_span = times[-1] - times[0]
    # 计算帧数，根据情况向最接近的整数四舍五入
    num_frames = round(total_time_span / min_non_zero_diff) + 1

    return TimeSeriesInfo(start_time, min_non_zero_diff, num_frames)


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    time_info: TimeSeriesInfo = TimeSeriesInfo(0, 0, 1)
    extra: dict = {}


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


from multiprocessing import Pool
from itertools import repeat

def _process_colmap_camera(key, cam_extrinsics, cam_intrinsics, images_folder, time_step):
    sys.stdout.write('\r')
    # the exact output you're looking for:
    sys.stdout.write("Reading camera {}/{}".format(key + 1, len(cam_extrinsics)))
    sys.stdout.flush()

    extr = cam_extrinsics[key]
    intr = cam_intrinsics[extr.camera_id]
    height = intr.height
    width = intr.width

    uid = intr.id
    R = np.transpose(qvec2rotmat(extr.qvec))
    T = np.array(extr.tvec)

    if intr.model == "SIMPLE_PINHOLE":
        focal_length_x = intr.params[0]
        FovY = focal2fov(focal_length_x, height)
        FovX = focal2fov(focal_length_x, width)
    elif intr.model == "PINHOLE":
        focal_length_x = intr.params[0]
        focal_length_y = intr.params[1]
        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)
    else:
        assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

    image_path = os.path.join(images_folder, extr.name)
    image_name = os.path.basename(image_path).split(".")[0]

    if time_step:
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=None,
                              image_path=image_path, image_name=image_name, width=width, height=height,
                              time=time_step * extr.frame_id)
    else:
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=None,
                              image_path=image_path, image_name=image_name, width=width, height=height, time=0)
    return cam_info

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, time_step=None):
    cam_infos = []

    # 创建进程池
    with Pool() as pool:
        # 使用pool.starmap并行处理
        results = pool.starmap(_process_colmap_camera,
                               zip(cam_extrinsics.keys(), repeat(cam_extrinsics), repeat(cam_intrinsics),
                                   repeat(images_folder), repeat(time_step)))
        cam_infos.extend(results)

    sys.stdout.write('\n')
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def _listFixedColmapSubFolder(path):
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
    numeric_dirs.sort(key=int)

    return numeric_dirs


def readFixedColmapInfo(path, eval, llffhold=8, time_step=1 / 30, timestep_x=1):
    cameras_views_file = os.path.join(path, "views.txt")
    cameras_images_file = os.path.join(path, "images.txt")
    cameras_cameras_file = os.path.join(path, "cameras.txt")
    cam_views = read_extrinsics_text(cameras_views_file)
    cam_images = read_extrinsics_text(cameras_images_file)
    cam_cameras = read_intrinsics_text(cameras_cameras_file)

    numeric_directories = _listFixedColmapSubFolder(path)

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
    cam_infos = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_cameras,
                                  images_folder=path, time_step=time_step)

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
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           time_info=time_info)
    return scene_info


def readColmapSceneInfo(path, images, eval, llffhold=8):
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

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                                           images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
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

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)
            time = frame.get('time', 0)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path,
                                        image_name=image_name, width=image.size[0], height=image.size[1], time=time))

    return cam_infos


def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)

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

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readNeurofluidInfo(path, white_background, eval, extension=".png", timestep_x=1):
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
            readCamerasFromTransforms(sub_path, "transforms_train.json", white_background, extension))
        test_cam_infos.extend(
            readCamerasFromTransforms(sub_path, "transforms_test.json", white_background, extension))

    time_info: TimeSeriesInfo = handle_time(train_cam_infos + test_cam_infos)
    if timestep_x != 1:
        time_info = TimeSeriesInfo(time_info.start_time, time_info.time_step * timestep_x,
                                   time_info.num_frames // timestep_x)

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

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           time_info=time_info,
                           extra={'box_info': box_info})
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "Neurofluid": readNeurofluidInfo,
    "FixedColmap": readFixedColmapInfo
}
