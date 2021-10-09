#ifndef _CUDA_UTIL_H
#define _CUDA_UTIL_H

#include <stdio.h>
#include <stdbool.h>
#include <cuda_runtime.h>

__host__ void
cuda_check(cudaError_t err, const char* file, const int line, bool fatal);

#define CUDA_CHECK(call) do { \
	cuda_check((call), __FILE__, __LINE__, 1); \
} while (0)

typedef struct cudaDeviceProp cudaDeviceProp;

__host__ void
check_gpu(const cudaDeviceProp* const dev_prop);

#endif
