#include <cuda_runtime.h>
#ifndef CUDA_SLEEP_HEADER
#define CUDA_SLEEP_HEADER

__device__ void kernelWithRandomSleep()
{
    // 每个线程有一个基于全局ID的种子
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int seed = (unsigned int)clock64() + globalId;

    // 使用种子生成随机数
    int sleepTime = seed % 100; // 随机等待时间
    // printf("%d\n", sleepTime);
    __nanosleep(sleepTime);
}
#endif