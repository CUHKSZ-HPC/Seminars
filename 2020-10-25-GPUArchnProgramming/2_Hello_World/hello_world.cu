extern "C" {

// This file explains how we index a thread globally

#include <stdio.h>
#include <cuda_runtime.h>
#include "cuda_util.h"


__global__ void 
helloworld_1DGrid1DBlock() {
    int global_threadid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello World from thread %d\n", global_threadid);
}

__global__ void
helloworld_1DGrid2DBlock() {
    int global_threadid = blockIdx.x * blockDim.x * blockDim.y 
        + threadIdx.y * blockDim.x + threadIdx.x;
    printf("Hello World from thread %d\n", global_threadid);
}

__global__ void
helloworld_1DGrid3DBlock() {
    int global_threadid = blockIdx.x * blockDim.x * blockDim.y * blockDim.z
        + threadIdx.z * blockDim.y * blockDim.x
        + threadIdx.y * blockDim.x + threadIdx.x;
    printf("Hello World from thread %d\n", global_threadid);
}

__global__ void
helloworld_2DGrid1DBlock() {
    int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    int global_threadid = blockId * blockDim.x + threadIdx.x;
    printf("Hello World from thread %d\n", global_threadid);
}

__global__ void
helloworld_2DGrid2DBlock() {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int global_threadid = blockId * (blockDim.x * blockDim.y)
        + (threadIdx.y * blockDim.x) + threadIdx.x;
    printf("Hello World from thread %d\n", global_threadid);
}

__global__ void
helloworld_2DGrid3DBlock() {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int global_threadid = blockId * (blockDim.x * blockDim.y * blockDim.z)
        + (threadIdx.z * blockDim.x * blockDim.y)
        + (threadIdx.y * blockDim.x) + threadIdx.x;
    printf("Hello World from thread %d\n", global_threadid);
}

__global__ void
helloworld_3DGrid1DBlock() {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x
		+ gridDim.x * gridDim.y * blockIdx.z;
	int global_threadid = blockId * blockDim.x + threadIdx.x;
    printf("Hello World from thread %d\n", global_threadid);
}

__global__ void
helloworld_3DGrid2DBlock() {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x
		+ gridDim.x * gridDim.y * blockIdx.z;
	int global_threadid = blockId * (blockDim.x * blockDim.y)
		+ (threadIdx.y * blockDim.x) + threadIdx.x;
	printf("Hello World from thread %d\n", global_threadid);
}

__global__ void
helloworld_3DGrid3DBlock() {
	int blockId = blockIdx.x + blockIdx.y * gridDim.x
		+ gridDim.x * gridDim.y * blockIdx.z;
	int global_threadid = blockId * (blockDim.x * blockDim.y * blockDim.z)
		+ (threadIdx.z * (blockDim.x * blockDim.y))
        + (threadIdx.y * blockDim.x) + threadIdx.x;
    printf("Hello World from thread %d\n", global_threadid);
}

int main() {
    printf("\n\n===== helloworld 1D Grid 1D Block =====\n");
    helloworld_1DGrid1DBlock<<<4, 4>>>();	
    CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("\n\n===== helloworld 1D Grid 2D Block =====\n");
    helloworld_1DGrid2DBlock<<<4, dim3(2, 2)>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("\n\n===== helloworld 1D Grid 3D Block =====\n");
    helloworld_1DGrid3DBlock<<<4, dim3(2, 2, 1)>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("\n\n===== helloworld 2D Grid 1D Block =====\n");
    helloworld_2DGrid1DBlock<<<dim3(2, 2), 4>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("\n\n===== helloworld 2D Grid 2D Block =====\n");
    helloworld_2DGrid2DBlock<<<dim3(2, 2), dim3(2, 2)>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("\n\n===== helloworld 2D Grid 2D Block =====\n");
    helloworld_2DGrid2DBlock<<<dim3(2, 2), dim3(2, 2)>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

	printf("\n\n===== helloworld 2D Grid 3D Block =====\n");
    helloworld_2DGrid3DBlock<<<dim3(2, 2), dim3(2, 2, 1)>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

	printf("\n\n===== helloworld 3D Grid 1D Block =====\n");
    helloworld_3DGrid1DBlock<<<dim3(2, 2), 4>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

	printf("\n\n===== helloworld 3D Grid 2D Block =====\n");
    helloworld_3DGrid2DBlock<<<dim3(2, 2, 1), dim3(2, 2)>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

	printf("\n\n===== helloworld 3D Grid 3D Block =====\n");
    helloworld_3DGrid3DBlock<<<dim3(2, 2, 1), dim3(2, 2, 1)>>>();
    CUDA_CHECK(cudaDeviceSynchronize());  
    
    return 0;
}
}
