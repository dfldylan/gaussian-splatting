#include <cuda_runtime.h>
#include <torch/extension.h>
#define BLOCK_SIZE 256

__device__ void mutexLock(int *mutex)
{
    while (atomicCAS(mutex, 0, 1) != 0)
        ;
}

__device__ void mutexUnlock(int *mutex)
{
    atomicExch(mutex, 0);
}

__device__ int _findLock(int *uf_array, int x, int *ball_lock)
{
    int root = x;
    mutexLock(&ball_lock[root]);
    while (uf_array[root] != root)
    {
        mutexUnlock(&ball_lock[root]);
        root = uf_array[root];
        mutexLock(&ball_lock[root]);
    }
    // 路径压缩
    while (uf_array[x] != x)
    {
        int tmp = uf_array[x];
        uf_array[x] = root;
        x = tmp;
    }
    return root;
}

__device__ int _find(int *uf_array, int x)
{
    int root = x;
    while (uf_array[root] != root)
    {
        root = uf_array[root];
    }
    // 路径压缩
    while (uf_array[x] != x)
    {
        int tmp = uf_array[x];
        uf_array[x] = root;
        x = tmp;
    }
    return root;
}

__device__ void _linkUf(
    int *uf_array,
    int a, // select
    int b, // neighbor
    int *ball_lock)
{
    int root_b = _find(uf_array, b);
    int root_a = _findLock(uf_array, a, ball_lock);
    uf_array[root_a] = root_b;
    mutexUnlock(&ball_lock[root_a]);
    uf_array[a] = root_b;
}

__global__ void initUfArray(
    int *uf_array,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;
    uf_array[idx] = idx;
}

__global__ void classifyBallUfCUDA(
    const float *center, // [N, 3]
    const float *radius, // [N]
    const float *color,  // [N, 3]
    const float dis_thr,
    const float color_thr,
    int n,
    int *uf_array,
    int *ball_lock)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;

    const float *new_center = center + idx * 3;
    float new_x = new_center[0];
    float new_y = new_center[1];
    float new_z = new_center[2];
    float new_radius = radius[idx];
    const float *new_color = color + idx * 3;
    float new_cx = new_color[0];
    float new_cy = new_color[1];
    float new_cz = new_color[2];
    float cthr2 = color_thr * color_thr;

    for (int k = 0; k < n; ++k)
    {
        float x = center[k * 3 + 0];
        float y = center[k * 3 + 1];
        float z = center[k * 3 + 2];
        float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) +
                   (new_z - z) * (new_z - z);
        float D = radius[k] + new_radius + dis_thr;
        float D2 = D * D;
        if (d2 < D2)
        {
            // judge color
            float cx = color[k * 3 + 0];
            float cy = color[k * 3 + 1];
            float cz = color[k * 3 + 2];
            float cd2 = (new_cx - cx) * (new_cx - cx) + (new_cy - cy) * (new_cy - cy) + (new_cz - cz) * (new_cz - cz);
            if (cd2 < cthr2)
            {
                // find one
                _linkUf(uf_array, idx, k, ball_lock);
            }
        }
    }
}

__global__ void ufToLabels(
    int *uf_array,
    int *out_labels,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;

    out_labels[idx] = _find(uf_array, idx);
}

torch::Tensor ClassifyBallCUDA(
    const torch::Tensor &center,
    const torch::Tensor &radius,
    const torch::Tensor &color,
    const float dis_thr,
    const float color_thr)
{
    int N = radius.size(0);
    int *uf_array;
    cudaMalloc(&uf_array, N * sizeof(int));
    int *ball_lock;
    cudaMalloc(&ball_lock, N * sizeof(int));
    cudaMemset(ball_lock, 0, N * sizeof(int));
    torch::Tensor out_label = torch::full({N}, 0.0, radius.options().dtype(torch::kInt32));

    int blockSize = BLOCK_SIZE;                      // 每个块的线程数
    int numBlocks = (N + blockSize - 1) / blockSize; // 计算所需的块数

    initUfArray<<<numBlocks, blockSize>>>(
        uf_array,
        N);

    classifyBallUfCUDA<<<numBlocks, blockSize>>>(
        center.contiguous().data_ptr<float>(),
        radius.contiguous().data_ptr<float>(),
        color.contiguous().data_ptr<float>(),
        dis_thr,
        color_thr,
        N,
        uf_array,
        ball_lock);

    ufToLabels<<<numBlocks, blockSize>>>(
        uf_array,
        out_label.contiguous().data_ptr<int>(),
        N);
    // 在CUDA内核调用后
    cudaFree(uf_array);
    cudaFree(ball_lock);
    return out_label;
}