#include <cuda_runtime.h>
#include <torch/extension.h>
#include "check_cuda.h"
#include "log.h"
#include "queue_cuda.cu"
#include "sleep.h"
#define BLOCK_SIZE 256
#define MAX_MATRIX 1000000000

__global__ void initUfArray(
    int *uf_array,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;
    uf_array[idx] = idx;
}

__global__ void calNeighbor(
    const float *center, // [N, 3]
    const float *radius, // [N]
    const float *color,  // [N, 3]
    const float dis_thr,
    const float color_thr,
    int N, int batchIndex, int batchSize,
    bool *neighbor_array)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;
    // producer
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

    for (int i = 0; i < batchSize; ++i)
    {
        int index = idx * batchSize + i;
        neighbor_array[index] = false;

        if (batchIndex * batchSize + i == int(N / 2))
            break;
        int k = (idx + batchIndex * batchSize + i + 1) % N;
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
                LOG(LOG_LEVEL_DEBUG, "gpu find pair %d %d", idx, k);
                neighbor_array[index] = true;
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
    int root = idx;
    while (uf_array[root] != root)
    {
        root = uf_array[root];
    }

    out_labels[idx] = root;
}
int findRootCpu(int *uf, int x)
{
    if (uf[x] != x)
    {
        uf[x] = findRootCpu(uf, uf[x]);
    }
    return uf[x];
}

void constructUf(int *uf, int idx, int k)
{
    int root_k = _findRoot(uf, k);
    int root_idx = _findRoot(uf, idx);
    if (root_k < root_idx)
    {
        uf[root_idx] = root_k;
    }
    else if (root_k > root_idx)
    {
        uf[root_k] = root_idx;
    }
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
    CHECK_CUDA(cudaMalloc(&uf_array, N * sizeof(int)));
    torch::Tensor out_label = torch::full({N}, 0.0, radius.options().dtype(torch::kInt32));

    int blockSize = BLOCK_SIZE;                      // 每个块的线程数
    int numBlocks = (N + blockSize - 1) / blockSize; // 计算所需的块数
    int maxMatrix = MAX_MATRIX;
    int batchSize = maxMatrix / N;

    initUfArray<<<numBlocks, blockSize>>>(
        uf_array,
        N);

    int *uf_cpu = (int *)malloc(sizeof(int) * N);
    CHECK_CUDA(cudaMemcpy(uf_cpu, uf_array, sizeof(int) * N, cudaMemcpyDeviceToHost));

    int loop_num = N / 2;
    bool *neighbor_array;
    CHECK_CUDA(cudaMalloc(&neighbor_array, N * batchSize * sizeof(bool)));

    bool *neighbor_array_cpu = (bool *)malloc(N * batchSize * sizeof(bool));

    int idx, i, inner_i;
    int batchIndex = 0;
    for (i = 0; i < loop_num; ++i)
    {
        inner_i = i - batchIndex * batchSize;
        if (inner_i >= 0)
        {
            calNeighbor<<<numBlocks, blockSize>>>(
                center.contiguous().data_ptr<float>(),
                radius.contiguous().data_ptr<float>(),
                color.contiguous().data_ptr<float>(),
                dis_thr,
                color_thr,
                N, batchIndex, batchSize,
                neighbor_array);
            batchIndex += 1;
            CHECK_CUDA(cudaMemcpy(neighbor_array_cpu, neighbor_array, N * batchSize * sizeof(bool), cudaMemcpyDeviceToHost));
        }
        else
        {
            inner_i += batchSize;
        }

        for (idx = 0; idx < N; ++idx)
        {
            if (neighbor_array_cpu[idx * batchSize + inner_i] == true)
            {
                int k = (idx + i + 1) % N;
                LOG(LOG_LEVEL_DEBUG, "cpu find pair %d %d", idx, k);
                constructUf(uf_cpu, idx, k);
            }
        }
    }

    CHECK_CUDA(cudaMemcpy(uf_array, uf_cpu, sizeof(int) * N, cudaMemcpyHostToDevice));
    ufToLabels<<<numBlocks, blockSize>>>(
        uf_array,
        out_label.contiguous().data_ptr<int>(),
        N);

    // cudaMemcpy(out_label.contiguous().data_ptr<int>(), uf_array, N * sizeof(int), cudaMemcpyDeviceToDevice);

    // 在CUDA内核调用后
    cudaFree(uf_array);
    cudaFree(neighbor_array);
    return out_label;
}
