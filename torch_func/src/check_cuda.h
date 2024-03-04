#ifndef CUDA_CHECK_HEADER
#define CUDA_CHECK_HEADER

#include "log.h"
#define CHECK_CUDA(A)                                                                                             \
    A;                                                                                                            \
    if (CURRENT_LOG_LEVEL != LOG_LEVEL_NONE)                                                                      \
    {                                                                                                             \
        auto ret = cudaDeviceSynchronize();                                                                       \
        if (ret != cudaSuccess)                                                                                   \
        {                                                                                                         \
            LOG(LOG_LEVEL_ERROR, "[CUDA ERROR] in %s\nLine %d: %s", __FILE__, __LINE__, cudaGetErrorString(ret)); \
            throw std::runtime_error(cudaGetErrorString(ret));                                                    \
        }                                                                                                         \
    }

#endif