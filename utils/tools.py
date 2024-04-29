import open3d as o3d
import numpy as np
import torch
from sklearn.cluster import DBSCAN

def generate_random_bool_tensor(size, n_true):
    assert n_true <= size, "n_true cannot be greater than the tensor size."

    # 创建一个全为 False 的 PyTorch 布尔张量
    result = torch.zeros(size, dtype=torch.bool)

    # 随机生成不重复的索引
    true_indices = torch.randperm(size)[:n_true]
    result[true_indices] = True

    return result

def classify_mask(xyz, eps=0.075, min_samples=10, first_class=0):
    # 使用DBSCAN进行聚类
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)  # eps和min_samples根据实际情况调整
    labels = dbscan.fit_predict(xyz)
    unique_labels, counts = np.unique(labels, return_counts=True)
    order = np.argsort(counts)[::-1]
    select_label = unique_labels[order[first_class]]
    return labels == select_label

def categorize(points, eps=0.05):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=10, print_progress=False))
    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
    order = np.argsort(counts)[::-1]
    return unique_labels[order], counts[order], labels


def crop_main(points, eps=0.05, choose=0):
    unique_labels, counts, labels = categorize(points, eps=eps)
    largest_cluster_label = unique_labels[choose]
    mask = labels == largest_cluster_label
    return mask


def similarity_mask(vectors, target, threshold=0.9, ret_fixed=False):
    if not isinstance(target, torch.Tensor):
        # Convert the list to a PyTorch tensor
        target = torch.tensor(target, dtype=torch.float32).cuda()

    # Calculate the Euclidean distance between each vector and the target
    bias = (vectors - target)
    dist2 = torch.sum(bias ** 2, axis=1)

    # Create a mask where distances are less than or equal to the threshold
    mask = dist2 <= threshold ** 2

    if ret_fixed:
        normal = bias / torch.sqrt(dist2).unsqueeze(-1)
        fixed = threshold * normal + target
        return mask, fixed

    return mask, None
