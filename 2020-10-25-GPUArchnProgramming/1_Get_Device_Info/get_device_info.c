#include <stdio.h>
#include "cuda_util.h"
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp dev_prop;
    CUDA_CHECK(cudaGetDeviceProperties(&dev_prop, 0));
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaDeviceReset());
    
    check_gpu(&dev_prop);

    return 0; 
}
