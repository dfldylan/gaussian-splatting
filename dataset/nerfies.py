import json
import logging
import multiprocessing
import os
from typing import List
from typing import Tuple  # Make sure this is imported if not already
from typing import Union, Optional  # Added List

import math  # Added for atan
import numpy as np

from dataset import ShootInfo


# Helper functions for undistortion (adapted from HyperNeRF)
def _compute_residual_and_jacobian(
        x: np.ndarray,
        y: np.ndarray,
        xd: np.ndarray,
        yd: np.ndarray,
        k1: float = 0.0,
        k2: float = 0.0,
        k3: float = 0.0,
        p1: float = 0.0,
        p2: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
np.ndarray]:
    """Auxiliary function of radial_and_tangential_undistort()."""
    # `r` is the squared radius from the lens center.
    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r2 * r4

    # `d` is the radial distortion factor.
    d_factor = 1.0 + k1 * r2 + k2 * r4 + k3 * r6

    # The distorted coordinates according to the model.
    # fx = d_factor*x + 2*p1*x*y + p2*(r2 + 2*x*x) - xd
    # fy = d_factor*y + 2*p2*x*y + p1*(r2 + 2*y*y) - yd

    # Corrected formulas for fx, fy based on typical distortion models
    # xd = x_distorted * (1 + k1*r^2 + k2*r^4 + k3*r^6) + (2*p1*x*y + p2*(r^2 + 2*x^2))
    # yd = y_distorted * (1 + k1*r^2 + k2*r^4 + k3*r^6) + (2*p2*x*y + p1*(r^2 + 2*y^2))
    # We are calculating residual: predicted_distorted - observed_distorted (xd, yd from input)
    # So, fx is the residual in x, fy is the residual in y.
    # x and y are current estimates of undistorted points.
    # xd and yd are the observed distorted points.

    # Predicted distorted x based on current estimate of undistorted x,y
    pred_xd = x * d_factor + (2 * p1 * x * y + p2 * (r2 + 2 * x ** 2))
    # Predicted distorted y based on current estimate of undistorted x,y
    pred_yd = y * d_factor + (2 * p2 * x * y + p1 * (r2 + 2 * y ** 2))

    fx = pred_xd - xd
    fy = pred_yd - yd

    # Derivatives of d_factor over x and y.
    # d(d_factor)/dr2 = k1 + 2*k2*r2 + 3*k3*r4
    # dr2/dx = 2x, dr2/dy = 2y
    d_d_factor_dr2 = k1 + 2 * k2 * r2 + 3 * k3 * r4

    d_d_factor_dx = 2 * x * d_d_factor_dr2
    d_d_factor_dy = 2 * y * d_d_factor_dr2

    # Derivatives of pred_xd over x and y
    # d(pred_xd)/dx = d_factor + x * d_d_factor_dx + (2*p1*y + p2*(2*x + 4*x))
    #               = d_factor + x * d_d_factor_dx + 2*p1*y + 6*p2*x
    fx_x = d_factor + x * d_d_factor_dx + 2 * p1 * y + p2 * (2 * x + 4 * x)  # simplified: 6*p2*x
    # d(pred_xd)/dy = x * d_d_factor_dy + (2*p1*x + p2*(2*y))
    #               = x * d_d_factor_dy + 2*p1*x + 2*p2*y
    fx_y = x * d_d_factor_dy + 2 * p1 * x + p2 * (2 * y)

    # Derivatives of pred_yd over x and y
    # d(pred_yd)/dx = y * d_d_factor_dx + (2*p2*y + p1*(2*x))
    #               = y * d_d_factor_dx + 2*p2*y + 2*p1*x
    fy_x = y * d_d_factor_dx + 2 * p2 * y + p1 * (2 * x)
    # d(pred_yd)/dy = d_factor + y * d_d_factor_dy + (2*p2*x + p1*(2*y + 4*y))
    #               = d_factor + y * d_d_factor_dy + 2*p2*x + 6*p1*y
    fy_y = d_factor + y * d_d_factor_dy + 2 * p2 * x + p1 * (2 * y + 4 * y)  # simplified: 6*p1*y

    return fx, fy, fx_x, fx_y, fy_x, fy_y


def _radial_and_tangential_undistort(
        xd: np.ndarray,
        yd: np.ndarray,
        k1: float = 0,
        k2: float = 0,
        k3: float = 0,
        p1: float = 0,
        p2: float = 0,
        eps: float = 1e-9,
        max_iterations=10) -> Tuple[np.ndarray, np.ndarray]:
    """Computes undistorted (x, y) from (xd, yd) using Newton's method."""
    # Initialize estimate of undistorted point with the distorted point.
    x = np.copy(xd)
    y = np.copy(yd)

    for _ in range(max_iterations):
        fx, fy, fx_x, fx_y, fy_x, fy_y = _compute_residual_and_jacobian(
            x=x, y=y, xd=xd, yd=yd, k1=k1, k2=k2, k3=k3, p1=p1, p2=p2)

        # Solve the linear system J * [dx; dy] = -[fx; fy]
        # J = [[fx_x, fx_y], [fy_x, fy_y]]
        # dx = (fx_y * fy - fx * fy_y) / (fx_x * fy_y - fx_y * fy_x)
        # dy = (fx_x * fy - fx * fy_x) / (fx_y * fy_x - fx_x * fy_y)
        #    = (fx * fy_x - fx_x * fy) / (fx_x * fy_y - fx_y * fy_x)

        denominator = fx_x * fy_y - fx_y * fy_x

        # Check for singularity
        use_update = np.abs(denominator) > eps

        # Compute step only where denominator is not too small
        # Note: original hypernerf had x_numerator = fx * fy_y - fy * fx_y
        # which corresponds to - (fy * fx_y - fx * fy_y)
        # For J * step = -residual:
        # step_x = (-fx * fy_y + fy * fx_y) / denominator
        # step_y = (-fy * fx_x + fx * fy_x) / denominator

        step_x = np.zeros_like(x)
        step_y = np.zeros_like(y)

        safe_denominator = np.where(use_update, denominator, np.ones_like(denominator))

        step_x_val = (fx_y * fy - fx * fy_y) / safe_denominator
        step_y_val = (
                                 fx * fy_x - fx_x * fy) / safe_denominator  # Corrected from (fy * fx_x - fx * fy_x) / denominator to match standard Newton update for J*step = -F

        step_x = np.where(use_update, step_x_val, step_x)
        step_y = np.where(use_update, step_y_val, step_y)

        x = x + step_x  # x_new = x_old - J_inv * F  => x_new = x_old + step
        y = y + step_y

        if not np.any(use_update) or (np.all(np.abs(step_x) < eps) and np.all(np.abs(step_y) < eps)):
            break
    return x, y


def load_scene_info(data_dir: str) -> Tuple[np.ndarray, float, float, float]:
    """Loads the scene center, scale, near and far from scene.json.

    Args:
      data_dir: the path to the dataset.

    Returns:
      scene_center: the center of the scene (unscaled coordinates).
      scene_scale: the scale of the scene.
      near: the near plane of the scene (scaled coordinates).
      far: the far plane of the scene (scaled coordinates).
    """
    scene_json_path = os.path.join(data_dir, "scene.json")
    with open(scene_json_path, "r") as f:
        scene_json = json.load(f)

    scene_center = np.array(scene_json["center"])
    scene_scale = scene_json["scale"]
    near = scene_json["near"]
    far = scene_json["far"]

    return scene_center, scene_scale, near, far


def load_camera(
        item_id: str,
        metadata_dict: dict,
        # keys are item_ids, values are metadata keys: {time_id, warp_id, appearance_id, camera_id}
        data_dir: str,  # Added type hint for clarity
        scene_center: Optional[np.ndarray] = None,
        scene_scale: Optional[Union[float, np.ndarray]] = None,
        image_scale: int = 1,  # Allow float for image_scale
        timestep: float = 1 / 30,  # Default to 30 FPS，
        depths_params: Optional[dict] = None,  # Optional parameter for depth parameters
        depths_folder: str = "",  # Optional parameter for depth folder
        seg_folder: str = "",  # Optional parameter for segmentation folder
) -> ShootInfo:
    """Loads camera parameters from a JSON file and creates a ShootInfo object.

    Args:
      camera_dir: The directory containing camera JSON files.
      item_id: The identifier for the camera (e.g., '000001').
      scene_center: The center of the scene to offset the camera position.
      scene_scale: The scale of the scene to apply to the camera position.
      image_scale: The factor by which the original image dimensions are scaled.
                   (e.g., image_scale=2 means images are 2x the original,
                    so camera parameters will be scaled by 0.5).

    Returns:
      A ShootInfo instance.
    """
    scale_factor = 1.0 / image_scale
    camera_dir = os.path.join(data_dir, "camera")
    camera_path = os.path.join(camera_dir, f"{item_id}.json")

    with open(camera_path, "r") as fp:
        camera_json = json.load(fp)

    # Fix for older camera JSON formats that might use "tangential"
    if "tangential" in camera_json and "tangential_distortion" not in camera_json:
        camera_json["tangential_distortion"] = camera_json["tangential"]

    # Extract original parameters from JSON
    R = np.asarray(camera_json['orientation'], dtype=np.float32)
    T = np.asarray(camera_json['position'], dtype=np.float32)
    focal_length_orig = float(camera_json['focal_length'])
    principal_point_orig = np.asarray(camera_json['principal_point'], dtype=np.float32)
    skew_orig = float(camera_json.get('skew', 0.0))  # Use .get for optional fields
    pixel_aspect_ratio_orig = float(camera_json.get('pixel_aspect_ratio', 1.0))
    radial_distortion_orig = np.asarray(camera_json.get('radial_distortion', [0.0, 0.0, 0.0]), dtype=np.float32)
    tangential_distortion_orig = np.asarray(camera_json.get('tangential_distortion', [0.0, 0.0]), dtype=np.float32)
    # image_size from JSON is typically [width, height]
    image_size = np.asarray(camera_json['image_size'], dtype=np.int32)

    # Apply scene transformations to position
    if scene_center is not None:
        T -= np.asarray(scene_center, dtype=np.float32)
    if scene_scale is not None:
        T *= np.asarray(scene_scale, dtype=np.float32)

    # Apply scale_factor to intrinsics for the scaled image
    fx_s = focal_length_orig * scale_factor

    width = int(round(image_size[0] * scale_factor))
    height = int(round(image_size[1] * scale_factor))

    # Calculate FoV for the scaled camera using pinhole camera model assumptions
    # FovX = 2 * atan(image_width / (2 * fx))
    # FovY = 2 * atan(image_height / (2 * fy))
    # fx = focal_length, fy = focal_length * pixel_aspect_ratio

    fy_s = fx_s * pixel_aspect_ratio_orig

    fov_x_rad = 2 * math.atan((width / 2.0) / fx_s)
    fov_y_rad = 2 * math.atan((height / 2.0) / fy_s)
    # Ensure the FoV is in radians
    # Attempt to form a relative path if camera_dir is deep enough
    image_folder_name = f'{int(image_scale)}x' if image_scale == int(image_scale) else f'{image_scale}x'
    image_path = os.path.join(data_dir, 'rgb', image_folder_name, f'{item_id}.png')  # Replaced with os.path.join

    depth_path = os.path.join(depths_folder, f"{item_id}.png") if depths_folder != "" else ""
    seg_path = os.path.join(seg_folder, f"{item_id}.png") if seg_folder != "" else ""
    depth_params: dict = {}
    if depths_params is not None:
        try:
            depth_params = depths_params[item_id]
        except:
            print("\n", item_id, "not found in depths_params")

    camera = ShootInfo(
        uid=int(item_id),
        R=R,
        T=T,
        FovY=fov_y_rad,
        FovX=fov_x_rad,
        depth_params=depth_params,
        image_path=image_path,
        image_name=item_id,
        depth_path=depth_path,
        seg_path=seg_path,
        width=width,
        height=height,
        time=timestep * int(metadata_dict[item_id]["time_id"]),  # Assuming time_id is in seconds
    )

    return camera


def _load_camera_mp(args):
    item_id, metadata_dict, data_dir, scene_center, scene_scale, image_scale, depths_params, depths_folder, seg_folder = args
    return load_camera(item_id, metadata_dict, data_dir, scene_center, scene_scale, image_scale, depths_params,
                       depths_folder, seg_folder)


def _load_dataset_ids(data_dir: str) -> Tuple[List[str], List[str]]:
    """Loads dataset IDs."""
    dataset_json_path = os.path.join(data_dir, "dataset.json")
    logging.info("*** Loading dataset IDs from %s", dataset_json_path)
    with open(dataset_json_path, "r") as f:
        dataset_json = json.load(f)
        train_ids = dataset_json["train_ids"]
        val_ids = dataset_json["val_ids"]

    train_ids = [str(i) for i in train_ids]
    val_ids = [str(i) for i in val_ids]

    return train_ids, val_ids


def readHypernerfCameras(item_ids, metadata_dict, data_dir, scene_center, scene_scale, image_scale,
                         depths_params: Optional[dict] = None,  # Optional parameter for depth parameters
                         depths_folder: str = "",  # Optional parameter for depth folder
                         seg_folder: str = "",  # Optional parameter for segmentation folder
                         ):
    # 多进程加载相机
    args_list = [
        (item_id, metadata_dict, data_dir, scene_center, scene_scale, image_scale, depths_params, depths_folder,
         seg_folder)
        for item_id in item_ids
    ]

    with multiprocessing.Pool() as pool:
        train_cam_infos = pool.map(_load_camera_mp, args_list)
    return train_cam_infos


def readNerfiesCameras(
        camera_path: str,
        scene_center: Optional[np.ndarray] = None,
        scene_scale: Optional[Union[float, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reads the Nerfies camera data from a JSON file.

    Args:
      camera_path: the path to the camera JSON file.
      scene_center: the center of the scene where the camera will be centered to.
      scene_scale: the scale of the scene by which the camera will also be scaled
        by.

    Returns:
      A tuple containing the camera positions and orientations.
    """
    with open(camera_path, "r") as f:
        cameras = json.load(f)

    positions = []
    orientations = []

    for cam in cameras:
        position = np.array(cam["position"])
        orientation = np.array(cam["orientation"])

        if scene_center is not None:
            position -= scene_center
        if scene_scale is not None:
            position *= scene_scale

        positions.append(position)
        orientations.append(orientation)

    return np.array(positions), np.array(orientations)
