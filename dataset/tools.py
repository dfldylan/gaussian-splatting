import json
import logging
from pathlib import Path
from typing import List

from plyfile import PlyData, PlyElement

from dataset import ShootInfo
from utils.graphics_utils import BasicPointCloud, getWorld2View2, focal2fov, fov2focal
from utils.sh_utils import SH2RGB
from utils.time_utils import TimeSeriesInfo


def fetchPly(path):
    plydata = PlyData.read(path)
    return loadPly(plydata)


def loadPly(plydata):
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    ply_data = buildPly(xyz, rgb)
    ply_data.write(path)


def buildPly(xyz, rgb):
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
    return PlyData([vertex_element])


def handle_time(cam_infos: List[ShootInfo]) -> TimeSeriesInfo:
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


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    cam: ShootInfo
    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


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

            cam_infos.append(ShootInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path,
                                       image_name=image_name, width=image.size[0], height=image.size[1], time=time))

    return cam_infos


def gen_random_points(num_pts=100_000, channel=3, edge_length=1.3):
    """
    Generates a random point cloud with color data based on the specified channel.

    Args:
        ply_path (str): Path to save the generated .ply file.
        num_pts (int): Number of points to generate in the cloud.
        channel (int): Number of color channels, 3 for RGB, 1 for grayscale. Default is 3.
    """
    # Generate random points
    logging.info(f"Generating random point cloud ({num_pts}) with {channel} channel(s)...")

    # We create random points inside the bounds of the synthetic Blender scenes
    xyz = (np.random.random((num_pts, 3)) * 2 - 1) * edge_length  # [-edge_length, edge_length) for each axis

    # Generate random colors according to the specified number of channels
    if channel == 3:
        shs = np.random.random((num_pts, 3)) / 255.0  # RGB color
        colors = SH2RGB(shs) * 255
    elif channel == 1:
        colors = np.ones((num_pts, 3)) * 255.0  # Grayscale color
    else:
        raise ValueError("channel must be 1 or 3")

    # Store the point cloud with colors
    return buildPly(xyz, colors)


from PIL import Image

# 7-tuple containing:
# pixel id x, pixel id y, ray start x, ray start y, ray direction x, ray direction y, ray direction z.
# def parse_rays_file_old(filepath: str):
#     """
#     Parses the rays.txt file to extract ray information.
#     """
#     with open(filepath, 'r') as file:
#         # Read the first line for width and height
#         width, height = map(int, file.readline().split())
#
#         # Use list comprehension to efficiently parse remaining lines
#         rays_data = [
#             list(map(float, line.split()))
#             for line in file
#             if len(line.split()) == 7
#         ]
#
#     # Convert to numpy array for better performance in numerical operations
#     return width, height, np.array(rays_data, dtype=np.float32)

import pandas as pd


def parse_rays_file(filepath: str):
    # 读取文件的第一行以获取宽度和高度
    with open(filepath, 'r') as file:
        first_line = file.readline().strip()
        width, height = map(int, first_line.split())

    # 使用Pandas读取剩余的数据
    rays_data = pd.read_csv(filepath, sep=' ', skiprows=1, header=None,
                            names=['pixel_id_x', 'pixel_id_y', 'ray_start_x',
                                   'ray_start_y', 'ray_direction_x',
                                   'ray_direction_y', 'ray_direction_z'],
                            engine='python')

    return width, height, rays_data


# def find_focal_point(rays_data):
#     """
#     Finds the focal point by calculating the intersection of all rays.
#     """
#     # Vectorized computation to avoid the for loop
#     ray_origins = rays_data[:, 2:5]
#     ray_origins[:, 2] = 0
#     ray_directions = rays_data[:, 4:]
#
#     # Precompute outer products for each ray direction
#     I = np.eye(3)
#     outer_products = np.einsum('ij,ik->ijk', ray_directions, ray_directions)
#     A_matrices = I - outer_products
#     b_vectors = np.einsum('ijk,ij->ik', A_matrices, ray_origins)
#
#     # Flatten A_matrices and b_vectors for least squares solving
#     A = A_matrices.reshape(-1, 3)
#     b = b_vectors.reshape(-1)
#
#     # Solve using least squares to find the focal point
#     focal_point, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
#
#     return focal_point
#


def find_focal_point(rays_data):
    """
    Finds the focal point by calculating the intersection of all rays.

    Parameters:
    - rays_data: Pandas DataFrame containing the ray data.

    Returns:
    - A numpy array representing the focal point coordinates.
    """
    # 提取光线起点和方向
    ray_origins = rays_data[['ray_start_x', 'ray_start_y']].to_numpy()
    ray_origins = np.column_stack((ray_origins, np.zeros(ray_origins.shape[0])))  # 添加z坐标为0
    ray_directions = rays_data[['ray_direction_x', 'ray_direction_y', 'ray_direction_z']].to_numpy()

    # 预计算每条光线方向的外积
    I = np.eye(3)
    outer_products = np.einsum('ij,ik->ijk', ray_directions, ray_directions)
    A_matrices = I - outer_products
    b_vectors = np.einsum('ijk,ij->ik', A_matrices, ray_origins)

    # 将A_matrices和b_vectors展平，以便进行最小二乘求解
    A = A_matrices.reshape(-1, 3)
    b = b_vectors.reshape(-1)

    # 使用最小二乘法求解焦点
    focal_point, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    return focal_point


# def get_ray_from_2d(rays_data, x, y):
#     """
#     Retrieve the ray direction vector for the given pixel coordinates (x, y).
#     """
#     ray = next((ray[4:7] for ray in rays_data if int(ray[1]) == y and int(ray[0]) == x), None)
#     if ray is None:
#         raise ValueError(f"Ray not found for pixel coordinates ({x}, {y})")
#     return np.array(ray)


def get_ray_from_2d(rays_data, x, y):
    """
    Retrieve the ray direction vector for the given pixel coordinates (x, y).

    Parameters:
    - rays_data: Pandas DataFrame containing the ray data.
    - x: Pixel coordinate x.
    - y: Pixel coordinate y.

    Returns:
    - A numpy array representing the ray direction vector.

    Raises:
    - ValueError if the ray is not found for the given pixel coordinates.
    """
    # 筛选出对应像素坐标的光线数据
    ray = rays_data[(rays_data['pixel_id_x'] == x) & (rays_data['pixel_id_y'] == y)]

    if ray.empty:
        raise ValueError(f"Ray not found for pixel coordinates ({x}, {y})")

    # 提取方向向量并转换为numpy数组
    return ray.iloc[0][['ray_direction_x', 'ray_direction_y', 'ray_direction_z']].to_numpy()


def compute_camera_info_from_rays(rays_data, width, height):
    """
    Given the rays data, compute rotation (R), translation (T), and field of view.
    """
    # Determine the center pixel coordinates
    center_x_2d = (width - 1) / 2
    center_y_2d = (height - 1) / 2

    # Get rays for left, right. handle displace rays
    for i in [(j // 2) * (-1) ** j for j in range(height)]:  # generate [0, 0, 1, -1, 2, -2, 3, -3, 4, -4, ...]
        try:
            left_ray = get_ray_from_2d(rays_data, 0, int(center_y_2d + i))
            right_ray = get_ray_from_2d(rays_data, width - 1, int(center_y_2d + i))
        except ValueError:
            logging.warning("get line rays failed at y_bias = {}".format(i))
            continue
        else:
            break

    # Get rays for top, and bottom edges
    for i in [(j // 2) * (-1) ** j for j in range(width)]:  # generate [0, 1, -1, 2, -2, 3, -3, 4, -4, ...]
        try:
            top_ray = get_ray_from_2d(rays_data, int(center_x_2d + i), 0)
            bottom_ray = get_ray_from_2d(rays_data, int(center_x_2d + i), height - 1)
        except ValueError:
            logging.warning("get row rays failed at x_bias = {}".format(i))
            continue
        else:
            break

    # Normalize the rays
    left_ray /= np.linalg.norm(left_ray)
    right_ray /= np.linalg.norm(right_ray)
    top_ray /= np.linalg.norm(top_ray)
    bottom_ray /= np.linalg.norm(bottom_ray)

    # Calculate x_axis and y_axis, coarse
    x_axis, y_axis = right_ray - left_ray, bottom_ray - top_ray
    x_axis /= np.linalg.norm(x_axis)
    y_axis /= np.linalg.norm(y_axis)

    # Calculate z_axis as the cross product of x_axis and y_axis
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)

    # new x_axis and y_axis
    y_axis_rotate_45 = x_axis + y_axis
    y_axis_rotate_45 /= np.linalg.norm(y_axis_rotate_45)
    x_axis_rotate_45 = np.cross(y_axis_rotate_45, z_axis)
    x_axis_rotate_45 /= np.linalg.norm(x_axis_rotate_45)
    x_axis = y_axis_rotate_45 + x_axis_rotate_45
    y_axis = y_axis_rotate_45 - x_axis_rotate_45
    x_axis /= np.linalg.norm(x_axis)
    y_axis /= np.linalg.norm(y_axis)

    # Create rotation matrix R
    R = np.vstack([x_axis, y_axis, z_axis]).T
    # Set translation vector T as the focal point
    T = find_focal_point(rays_data)

    # Calculate FoV in X direction (fovx) using the rays at the middle row's left and right ends
    fovx = np.arccos(np.clip(np.dot(left_ray, right_ray), -1.0, 1.0))

    # Calculate FoV in Y direction (fovy) using the rays at the middle column's top and bottom ends
    fovy = np.arccos(np.clip(np.dot(top_ray, bottom_ray), -1.0, 1.0))

    # c2w
    return R, T, fovy, fovx


def rays_txt_to_camera(rays_filepath):
    try:
        width, height, rays_data = parse_rays_file(rays_filepath)
        R, T, fovy, fovx = compute_camera_info_from_rays(rays_data, width, height)
    except ValueError:
        raise ValueError(f"Invalid rays file: {rays_filepath}")
    return width, height, R, T, fovy, fovx


def readCamerasFromScalarFlow(base_path: str, calib_folder, bg_threshold=8):
    """
    Generate a list of CameraInfo objects based on rays calibration data.
    """
    cam_infos = []
    json_file_path = os.path.join(base_path, "cameras.json")

    if os.path.exists(json_file_path):
        # Load camera data from JSON file
        data = json.load(open(json_file_path, 'r'))
        for cam_data in data:
            id = cam_data['id']
            width = cam_data['width']
            height = cam_data['height']
            R = np.array(cam_data['rotation'])
            T = np.array(cam_data['position'])
            fovy = cam_data['fy']
            fovx = cam_data['fx']
            img_name = cam_data['img_name']
            time = cam_data['time']

            image_path = os.path.join(base_path, img_name)
            image = Image.open(image_path)
            cam_infos.append(
                ShootInfo(uid=id, R=R, T=-R.T @ T, FovY=fovy, FovX=fovx, image=image, image_path=image_path,
                          image_name=img_name, width=width, height=height, time=time))
    else:
        # Read camera data from txt files and save to JSON if cache is enabled
        camera_infos = []
        for idx in range(1, 6):  # Assuming we have files named 1_rays.txt, 2_rays.txt, etc.
            rays_filepath = os.path.join(calib_folder, f"{idx}_rays.txt")
            width, height, R, T, fovy, fovx = rays_txt_to_camera(rays_filepath)
            camera_infos.append((width, height, R, T, fovy, fovx))

        os.makedirs(os.path.join(base_path, "images"), exist_ok=True)

        # Load image data
        cameras_data = []
        bg = []
        for idx in range(len(os.listdir(os.path.join(base_path, 'input', 'cam')))):
            # Set dummy image data
            npz_path = os.path.join(base_path, 'input', 'cam', f"imgsUnproc_{idx:06}.npz")
            logging.debug("Handling {}".format(npz_path))
            if not os.path.exists(npz_path):
                break

            npy = np.load(npz_path)['data']
            for i in range(5):
                arr = npy[i, :, :, 0]
                # 翻转数组，使得图像的上下方向正确
                arr = np.flipud(arr)
                image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "L")

                if idx == 0:
                    # 对于idx为0的情况，保存背景图像
                    bg.append(np.array(image))

                # 将当前图像转换为NumPy数组
                image_arr = np.array(image)
                bg_average = bg[i]
                # 在计算差异之前，先对背景图像和当前图像应用高斯模糊
                # bg_blurred = cv2.GaussianBlur(bg_average, gaussian_blur_kernel, 0)
                # image_arr_blurred = cv2.GaussianBlur(image_arr, gaussian_blur_kernel, 0)
                # diff = np.abs(image_arr_blurred - bg_blurred)
                diff = np.abs(image_arr.astype(np.int16) - bg_average.astype(np.int16))
                # 创建前景掩码，差异大于阈值的部分保留为前景
                foreground_mask = diff > bg_threshold
                # 创建去除背景的图像，背景部分设为0
                image_no_bg_arr = np.where(foreground_mask, image_arr, 0)
                # 将结果转换为图像
                image = Image.fromarray(image_no_bg_arr.astype(np.byte), "L").convert(mode="RGB")
                image_name = os.path.join("images", f"{i + 1}_{idx:04}.png")
                image_path = os.path.join(base_path, image_name)
                image.save(open(image_path, "wb")) if not os.path.exists(image_path) else None
                (width, height, R, T, _, _) = camera_infos[i]
                fovy, fovx = camera_infos[0][-2], camera_infos[0][-1]
                uid = idx * 10 + i + 1
                time = idx / 60
                cameras_data.append({
                    "id": uid,
                    "img_name": image_name,
                    "width": width,
                    "height": height,
                    "rotation": R.tolist(),
                    "position": T.tolist(),
                    "fy": fovy,
                    "fx": fovx,
                    "time": time,
                })
                cam_infos.append(
                    ShootInfo(uid=uid, R=R, T=-R.T @ T, FovY=fovy, FovX=fovx, image=image, image_path=image_path,
                              image_name=image_name, width=width, height=height, time=time))

        json.dump(cameras_data, open(json_file_path, 'w'), indent=4)

    return cam_infos


import os
import numpy as np
