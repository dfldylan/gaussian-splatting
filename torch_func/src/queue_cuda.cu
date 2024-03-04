#include <cuda_runtime.h>
#include "check_cuda.h"
#include "lock.cu"
#include "sleep.h"
#define QUEUE_SIZE 512

struct Queue
{
    int *data;
    int *head;
    int *count;
    int *lock;
};

void initQueue(Queue *queue, int N)
{
    CHECK_CUDA(cudaMalloc(&(queue->data), QUEUE_SIZE * N * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&(queue->head), N * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&(queue->count), N * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&(queue->lock), N * sizeof(int)));

    CHECK_CUDA(cudaMemset(queue->data, 0, QUEUE_SIZE * N * sizeof(int)));
    CHECK_CUDA(cudaMemset(queue->head, 0, N * sizeof(int)));
    CHECK_CUDA(cudaMemset(queue->count, 0, N * sizeof(int)));
    CHECK_CUDA(cudaMemset(queue->lock, 0, N * sizeof(int)));
}
void freeQueue(Queue *queue)
{
    cudaFree(queue->data);
    cudaFree(queue->head);
    cudaFree(queue->count);
    cudaFree(queue->lock);
}

__device__ void _pushQueue(Queue *queue, int idx, int k)
{
    while (true)
    {
        __nanosleep(100);
        int globalId = blockIdx.x * blockDim.x + threadIdx.x;
        mutexLock(&(queue->lock[idx]));
        // printf("T%d success to lock %d\n", globalId, idx);

        if (queue->count[idx] == QUEUE_SIZE)
        {
            mutexUnlock(&(queue->lock[idx]));
            // printf("1\n");
            // kernelWithRandomSleep();
        }
        else
        {
            break;
        }
    }
    if (queue->count[idx] == 0 || queue->data[idx * QUEUE_SIZE + (queue->head[idx] + queue->count[idx] - 1) % QUEUE_SIZE] != k)
    {
        queue->data[idx * QUEUE_SIZE + (queue->head[idx] + queue->count[idx]) % QUEUE_SIZE] = k;
        queue->count[idx] += 1;
    }
    mutexUnlock(&(queue->lock[idx]));
}

__device__ int _popQueue(Queue *queue, int idx)
{
    // printf("pop\n");
    if (queue->count[idx] == 0)
    {
        // mutexUnlock(&(queue->lock[idx]));
        return -1;
    }
    else
    {
        mutexLock(&(queue->lock[idx]));
    }

    int ret = queue->data[idx * QUEUE_SIZE + queue->head[idx]];
    queue->count[idx] -= 1;
    queue->head[idx] = (queue->head[idx] + 1) % QUEUE_SIZE;
    mutexUnlock(&(queue->lock[idx]));
    return ret;
}

__host__ __device__  int _findRoot(
    int *uf_array,
    int x)
{
    int root = x;
    while (uf_array[root] != root)
    {
        root = uf_array[root];
    }
    return root;
}
