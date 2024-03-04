#ifndef CUDA_LOG_HEADER
#define CUDA_LOG_HEADER

#include <stdio.h>

// 定义当前的日志等级
enum LogLevel
{
    LOG_LEVEL_DEBUG,   // 详细调试信息
    LOG_LEVEL_INFO,    // 一般信息
    LOG_LEVEL_WARNING, // 警告信息
    LOG_LEVEL_ERROR,   // 错误信息
    LOG_LEVEL_NONE     // 不输出任何日志
};
#define CURRENT_LOG_LEVEL LOG_LEVEL_NONE

// 日志宏，只有当当前日志等级小于或等于指定的日志等级时才输出
#define LOG(level, format, ...)                                 \
    do                                                          \
    {                                                           \
        if (level >= CURRENT_LOG_LEVEL)                         \
        {                                                       \
            printf("[%s] " format "\n", #level, ##__VA_ARGS__); \
        }                                                       \
    } while (0)

#endif