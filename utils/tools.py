import torch


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


if __name__ == '__main__':
    # 示例
    size = 10
    n_true = 3
    random_tensor = generate_random_bool_tensor(size, n_true)
    print(random_tensor)
