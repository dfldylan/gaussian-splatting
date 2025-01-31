import json
import struct
import numpy as np
import os
from dataset.colmap import read_next_bytes,CAMERA_MODEL_IDS,CAMERA_MODEL_NAMES


def read_camera_bin_to_json(camera_bin_path, output_json_path):
    """
    Reads camera.bin and exports it to a JSON file.
    """
    cameras = {}
    with open(camera_bin_path, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, 24, "iiQQ")
            camera_id, model_id, width, height = camera_properties
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8 * num_params,
                                     format_char_sequence="d" * num_params)
            cameras[camera_id] = {
                "id": camera_id,
                "model": model_name,
                "width": width,
                "height": height,
                "params": params
            }
    with open(output_json_path, "w") as json_file:
        json.dump(cameras, json_file, indent=4)
    print(f"Cameras exported to {output_json_path}")


def read_image_bin_to_json(image_bin_path, output_json_path):
    """
    Reads image.bin and exports it to a JSON file.
    """
    images = {}
    with open(image_bin_path, "rb") as fid:
        num_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_images):
            image_properties = read_next_bytes(fid, 64, "idddddddi")
            image_id = image_properties[0]
            qvec = image_properties[1:5]
            tvec = image_properties[5:8]
            camera_id = image_properties[8]

            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]

            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            x_y_id_s = read_next_bytes(fid, num_points2D * 24, "ddq" * num_points2D)
            xys = np.column_stack([x_y_id_s[0::3], x_y_id_s[1::3]])
            point3D_ids = x_y_id_s[2::3]

            images[image_id] = {
                "id": image_id,
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": image_name
            }
    with open(output_json_path, "w") as json_file:
        json.dump(images, json_file, indent=4)
    print(f"Images exported to {output_json_path}")



import json
import numpy as np
from scipy.spatial.transform import Rotation as R


def create_transforms_json(image_json_path, camera_json_path, output_json_path,time_step=1/30):
    # 读取输入JSON文件
    with open(image_json_path, "r") as f:
        image_data = json.load(f)
    with open(camera_json_path, "r") as f:
        camera_data = json.load(f)

    # 获取相机参数
    camera_angle_x = 2 * np.arctan(camera_data["1"]["width"] / (2 * camera_data["1"]["params"][0]))

    transforms = {
        "init_particle": "init_particle.npz",
        "camera_angle_x": camera_angle_x,
        "bounding_box": "box.pt",
        "frames": []
    }

    sorted_keys = sorted(image_data.keys(), key=lambda x: int(x))

    for image_id in sorted_keys:
        image_info = image_data[image_id]
        # 获取四元数和位移向量
        qvec = image_info["qvec"]
        tvec = image_info["tvec"]

        # 计算旋转矩阵
        rotation = R.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]])  # 注意四元数顺序
        rotation_matrix = rotation.as_matrix()

        # 构建变换矩阵
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix
        transform_matrix[:3, 3] = tvec

        # 构造单帧数据
        frame = {
            "file_path": f"train/{int(image_info['name'].split('.')[0]):04d}",
            "time": time_step * int(image_info['name'].split('.')[0]),
            "transform_matrix": transform_matrix.tolist(),
            "particle_path": f"particles/{int(image_info['name'].split('.')[0]):04d}.npz"
        }

        transforms["frames"].append(frame)

    # 保存到transforms.json
    with open(output_json_path, "w") as f:
        json.dump(transforms, f, indent=4)
    print(f"Transforms JSON saved to {output_json_path}")



if __name__ == '__main__':
    # read_camera_bin_to_json("/workdir/datasets/lava/sparse/0/cameras.bin", "output_camera.json")
    # read_image_bin_to_json("/workdir/datasets/lava/sparse/0/images.bin", "output_image.json")
    # 使用示例
    create_transforms_json(
        "output_image.json",
        "output_camera.json",
        "transforms.json"
    )
