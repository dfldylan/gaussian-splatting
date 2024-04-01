import torch
import numpy as np
from sklearn.cluster import DBSCAN


def generate_random_bool_tensor(size, n_true):
    assert n_true <= size, "n_true cannot be greater than the tensor size."

    # 创建一个全为 False 的 PyTorch 布尔张量
    result = torch.zeros(size, dtype=torch.bool)

    # 随机生成不重复的索引
    true_indices = torch.randperm(size)[:n_true]
    result[true_indices] = True

    return result


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


def classify_mask(xyz, eps=0.075, min_samples=10, first_class=0):
    # 使用DBSCAN进行聚类
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)  # eps和min_samples根据实际情况调整
    labels = dbscan.fit_predict(xyz)
    unique_labels, counts = np.unique(labels, return_counts=True)
    order = np.argsort(counts)[::-1]
    select_label = unique_labels[order[first_class]]
    return labels == select_label


if __name__ == '__main__':
    # 假设你已经有了一些PyTorch张量数据
    # 这里我们创建一个随机张量作为示例
    data_tensor = torch.rand(100, 2)  # 100个样本，每个样本2个特征

    # 将PyTorch张量转换为NumPy数组
    data_numpy = data_tensor.numpy()

    # 使用DBSCAN进行聚类
    dbscan = DBSCAN(eps=0.1, min_samples=10)  # eps和min_samples根据实际情况调整
    clusters = dbscan.fit_predict(data_numpy)

    # clusters变量现在包含每个数据点的簇标签
    print(clusters)
