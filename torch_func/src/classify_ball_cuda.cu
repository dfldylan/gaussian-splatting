#include <cuda_runtime.h>
#include <torch/extension.h>
#define BLOCK_SIZE 256

__device__ void mutexLock(int *mutex)
{
    while (atomicCAS(mutex, 0, 1) != 0)
    {
        __nanosleep(10);
    }
}

__device__ bool mutexTryLock(int *mutex)
{
    if (atomicCAS(mutex, 0, 1) == 0)
    {
        return true;
    }
    else
    {
        return false;
    }
}

__device__ void mutexUnlock(int *mutex)
{
    atomicExch(mutex, 0);
}

__device__ void _linkUf(
    int *uf_array,
    int a, // select
    int b, // neighbor
    int *ball_lock)
{
    // get and lock root_a and root_b
    int root_a, root_b, x;
    bool finish = false;
    bool failed = false;
    while (true)
    {
        __nanosleep((a % 10) * 10);
        failed = false;
        root_a = a;
        mutexLock(&ball_lock[root_a]);
        while (uf_array[root_a] != root_a)
        {
            mutexUnlock(&ball_lock[root_a]);
            root_a = uf_array[root_a];
            if (!mutexTryLock(&ball_lock[root_a]))
            {
                failed = true;
                break;
            };
        }
        if (failed)
        {
            continue;
        }
        // get and lock root_a success
        // 路径压缩
        x = a;
        while (uf_array[x] != root_a)
        {
            int tmp = uf_array[x];
            uf_array[x] = root_a;
            //printf("idx %d link %d -> %d\n", a, x, root_a);
            x = tmp;
        }

        root_b = b;
        if (root_a == root_b)
        {
            finish = true;
            mutexUnlock(&ball_lock[root_a]);
            break;
        }
        if (!mutexTryLock(&ball_lock[root_b]))
        {
            failed = true;
            mutexUnlock(&ball_lock[root_a]);
            continue;
        }
        while (uf_array[root_b] != root_b)
        {
            mutexUnlock(&ball_lock[root_b]);
            root_b = uf_array[root_b];
            if (root_a == root_b)
            {
                finish = true;
                break;
            }
            if (!mutexTryLock(&ball_lock[root_b]))
            {
                failed = true;
                break;
            };
        }
        if (finish)
        {
            mutexUnlock(&ball_lock[root_a]);
            break;
        }
        if (failed)
        {
            mutexUnlock(&ball_lock[root_a]);
            continue;
        }
        // get and lock root_b success
        // 路径压缩
        x = b;
        while (uf_array[x] != root_b)
        {
            int tmp = uf_array[x];
            uf_array[x] = root_b;
            //printf("idx %d link %d -> %d\n", a, x, root_b);
            x = tmp;
        }

        uf_array[a] = root_b;
        uf_array[root_a] = root_b;
        //printf("idx %d link %d -> %d\n", a, a, root_b);
        //printf("idx %d link %d -> %d\n", a, root_a, root_b);

        mutexUnlock(&ball_lock[root_a]);
        mutexUnlock(&ball_lock[root_b]);
        break;
    }
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

    int loop_num = n / 2;
    for (int i = 0; i < loop_num; ++i)
    {
        int k = (idx - 1 - i) % n;
        // printf("idx %d start handle %d\n", idx, k);
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
                //printf("idx %d find neighbor %d\n", idx, k);
                _linkUf(uf_array, idx, k, ball_lock);
                //printf("idx %d linked neighbor %d\n", idx, k);
            }
        }
        // printf("idx %d have handled %d\n", idx, k);
    }
    //printf("idx %d is finished!\n", idx);
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

    // cudaMemcpy(out_label.contiguous().data_ptr<int>(), uf_array, N * sizeof(int), cudaMemcpyDeviceToDevice);

    // 在CUDA内核调用后
    cudaFree(uf_array);
    cudaFree(ball_lock);
    return out_label;
}