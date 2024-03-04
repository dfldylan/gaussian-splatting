#include <cuda_runtime.h>
#include <torch/extension.h>
#include "check_cuda.h"
#include "log.h"
#include "queue_cuda.cu"
#include "sleep.h"
#define BLOCK_SIZE 256


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
    Queue _queue,
    bool *_finishFlags)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
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

    for (int k = 0; k < idx; ++k)
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
                int root_k = _findRoot(uf_array, k);
                if (uf_array[idx] == idx)
                {
                    uf_array[idx] = root_k;
                    LOG(LOG_LEVEL_INFO, "T%d linked %d to %d which is root of %d", idx, idx, root_k, k);
                }
                else if (uf_array[idx] > root_k)
                {
                    LOG(LOG_LEVEL_DEBUG, "T%d want to push %d which is root of %d to %d", idx, root_k, k, uf_array[idx]);
                    _pushQueue(&_queue, uf_array[idx], root_k);
                    LOG(LOG_LEVEL_INFO, "T%d have push %d which is root of %d to %d", idx, root_k, k, uf_array[idx]);
                    uf_array[idx] = root_k;
                    LOG(LOG_LEVEL_INFO, "T%d linked %d to %d which is root of %d", idx, idx, root_k, k);
                }
                else if (uf_array[idx] < root_k)
                {
                    LOG(LOG_LEVEL_DEBUG, "T%d want to push %d to %d which is root of %d", idx, uf_array[idx], root_k, k);
                    _pushQueue(&_queue, root_k, uf_array[idx]);
                    LOG(LOG_LEVEL_INFO, "T%d have push %d to %d which is root of %d", idx, uf_array[idx], root_k, k);
                }
            }
        }
    }

    // customer
    while (true)
    {
        bool finishFlag;
        if (idx == n - 1)
            finishFlag = true;
        else if (_finishFlags[idx + 1] == true)
        {
            finishFlag = true;
        }
        else
            finishFlag = false;

        // __nanosleep(1000);
        LOG(LOG_LEVEL_DEBUG, "T%d want to pop", idx);
        int k = _popQueue(&_queue, idx);
        if (k != -1)
            LOG(LOG_LEVEL_INFO, "T%d pop %d", idx, k);
        if (k == idx)
            continue;
        else if (k > idx)
            LOG(LOG_LEVEL_ERROR, "T%d queue have item %d large than self", idx, k);
        else if (k != -1)
        {
            int root_k = _findRoot(uf_array, k);
            if (root_k > k)
                LOG(LOG_LEVEL_ERROR, "T%d root_k %d is large than k %d", idx, root_k, k);

            if (uf_array[idx] == idx)
            {
                uf_array[idx] = root_k;
                LOG(LOG_LEVEL_INFO, "T%d linked %d to %d which is root of %d", idx, idx, root_k, k);
            }
            else if (uf_array[idx] > root_k)
            {
                LOG(LOG_LEVEL_DEBUG, "T%d want to push %d which is root of %d to %d", idx, root_k, k, uf_array[idx]);
                _pushQueue(&_queue, uf_array[idx], root_k);
                LOG(LOG_LEVEL_INFO, "T%d have push %d which is root of %d to %d", idx, root_k, k, uf_array[idx]);
                uf_array[idx] = root_k;
                LOG(LOG_LEVEL_INFO, "T%d linked %d to %d which is root of %d", idx, idx, root_k, k);
            }
            else if (uf_array[idx] < root_k)
            {
                LOG(LOG_LEVEL_DEBUG, "T%d want to push %d to %d which is root of %d", idx, uf_array[idx], root_k, k);
                _pushQueue(&_queue, root_k, uf_array[idx]);
                LOG(LOG_LEVEL_INFO, "T%d have push %d to %d which is root of %d", idx, uf_array[idx], root_k, k);
            }
        }
        else if (finishFlag == false)
        {
            continue;
        }
        else
        {
            _finishFlags[idx] = true;
            LOG(LOG_LEVEL_INFO, "T%d finished", idx);
            break;
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
    Queue queue;
    initQueue(&queue, N);
    bool *_flagArray;
    CHECK_CUDA(cudaMalloc(&_flagArray, N * sizeof(bool)));
    CHECK_CUDA(cudaMemset(_flagArray, false, N * sizeof(bool)));

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
        queue,
        _flagArray);

    ufToLabels<<<numBlocks, blockSize>>>(
        uf_array,
        out_label.contiguous().data_ptr<int>(),
        N);

    // cudaMemcpy(out_label.contiguous().data_ptr<int>(), uf_array, N * sizeof(int), cudaMemcpyDeviceToDevice);

    // 在CUDA内核调用后
    cudaFree(uf_array);
    cudaFree(_flagArray);
    freeQueue(&queue);
    return out_label;
}