#include <cuda_runtime.h>
#include "sleep.h"

__device__ void mutexLock(int *mutex)
{
    while (atomicCAS(mutex, 0, 1) != 0)
    {
        int globalId = blockIdx.x * blockDim.x + threadIdx.x;
        // printf("T%d lockfail\n", globalId);
        // kernelWithRandomSleep();
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
