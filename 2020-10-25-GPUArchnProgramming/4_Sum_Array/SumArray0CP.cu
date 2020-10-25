#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

void checkResult(float *hostRef, float *gpuRef, const int N) {
    double epsilon = 1.0E-8;
    bool match = 1;
    for (int i=0; i<N; i++) {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n",hostRef[i],gpuRef[i],i);
            break;
        }
    }
    if (match) printf("Arrays match.\n\n");
}

void initialData(float *ip,int size) {
    // generate different seed for random number
    time_t t;
    srand((unsigned) time(&t));
    for (int i=0; i<size; i++) {
        ip[i] = (float)( rand() & 0xFF )/10.0f;
    }
}

void sumArraysOnHost(float *A, float *B, float *C, const int N) {
    for (int idx=0; idx<N; idx++)
        C[idx] = A[idx] + B[idx];
}

__global__ void sumArraysZeroCopy(float *A, float *B, float *C, const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

int main(int argc, char **argv) {
    // part 0: set up device and array
    // set up device
    int dev = 0;
    cudaSetDevice(dev);
    
    // get device properties
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    
    // check if support mapped memory
    if (!deviceProp.canMapHostMemory) {
        printf("Device %d does not support mapping CPU host memory!\n", dev);
        cudaDeviceReset();
        exit(EXIT_SUCCESS);
    }
    printf("Using Device %d: %s ", dev, deviceProp.name);
    
    // set up date size of vectors
    int ipower = 10;
    if (argc>1) ipower = atoi(argv[1]);
    int nElem = 1<<ipower;
    size_t nBytes = nElem * sizeof(float);
    if (ipower < 18) {
        printf("Vector size %d power %d nbytes %3.0f KB\n", nElem,\
        ipower,(float)nBytes/(1024.0f));
    } else {
        printf("Vector size %d power %d nbytes %3.0f MB\n", nElem,\
        ipower,(float)nBytes/(1024.0f*1024.0f));
    }
    
    // part 1: using device memory
    // malloc host memory
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);
    
    // initialize data at host side
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);
    
    // add vector at host side for result checks
    sumArraysOnHost(h_A, h_B, hostRef, nElem);
    
    // malloc device global memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((float**)&d_A, nBytes);
    cudaMalloc((float**)&d_B, nBytes);
    cudaMalloc((float**)&d_C, nBytes);
    
    // transfer data from host to device
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);
    
    // set up execution configuration
    int iLen = 512;
    dim3 block (iLen);
    dim3 grid ((nElem+block.x-1)/block.x);
    
    // invoke kernel at host side
    sumArrays <<<grid, block>>>(d_A, d_B, d_C, nElem);
    
    // copy kernel result back to host side
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
    
    // check device results
    checkResult(hostRef, gpuRef, nElem);
    
    // free device global memory
    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);
    free(h_B);
    
    // part 2: using zerocopy memory for array A and B
    // allocate zerocpy memory
    unsigned int flags = cudaHostAllocMapped;
    cudaHostAlloc((void **)&h_A, nBytes, flags);
    cudaHostAlloc((void **)&h_B, nBytes, flags);
    
    // initialize data at host side
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);
    
    // pass the pointer to device
    cudaHostGetDevicePointer((void **)&d_A, (void *)h_A, 0);
    cudaHostGetDevicePointer((void **)&d_B, (void *)h_B, 0);
    
    // add at host side for result checks
    sumArraysOnHost(h_A, h_B, hostRef, nElem);
    
    // execute kernel with zero copy memory
    sumArraysZeroCopy <<<grid, block>>>(d_A, d_B, d_C, nElem);
    
    // copy kernel result back to host side
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
    
    // check device results
    checkResult(hostRef, gpuRef, nElem);
    
    // free memory
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    free(hostRef);
    free(gpuRef);
    
    // reset device
    cudaDeviceReset();
    return EXIT_SUCCESS;
}