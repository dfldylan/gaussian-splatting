import open3d as o3d
import numpy as np
import torch

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
    distances = torch.sqrt(torch.sum(bias ** 2, axis=1))

    # Create a mask where distances are less than or equal to the threshold
    mask = distances <= threshold

    if ret_fixed:
        normal = bias / distances.unsqueeze(-1)
        fixed = threshold * normal + target
        return mask, fixed

    return mask, None
