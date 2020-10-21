extern "C" {

#include "cuda_util.h"
	
__host__ void
cuda_check(cudaError_t err, const char* file, const int line, bool fatal) {
    if (cudaSuccess != err) {
        printf("[!] GPU error: %s:%d, code: %d, reason: %s\n",
			file, line, err, cudaGetErrorString(err));

	if (fatal) {
	    printf("[!] aborting...\n");
	    exit(err);
	}
    }
}

__host__ void
check_gpu(const cudaDeviceProp* const dev_prop) {

    printf("[+] CUDA info:\n");
    int cuda_rtv;
    CUDA_CHECK(cudaRuntimeGetVersion(&cuda_rtv));
    printf("\t\tRuntime version: %d.%d\n", cuda_rtv / 1000,
                 (cuda_rtv % 100) / 10);

    printf("\t\tCapability: %d.%d\n", dev_prop->major, dev_prop->minor);

    printf("\t\tGlobal memory size: %d Mbytes\n", dev_prop->totalGlobalMem / (1024*1024));

    printf("\t\tGPU clock rate: %0.2f MHz\n", dev_prop->clockRate / 1000.0f);
    printf("\t\tMemory clock rate: %0.2f MHz\n",
                 dev_prop->memoryClockRate / 1000.0f);
    printf("\t\tMemory bus width: %d bits\n", dev_prop->memoryBusWidth);
    printf("\t\tMax memory pitch: %d bytes\n",
                 dev_prop->memPitch);
    printf("\t\tL1 support global cache? %s\n",
                 dev_prop->globalL1CacheSupported ? "yes" : "no");
    printf("\t\tL1 support local cache? %s\n",
                 dev_prop->localL1CacheSupported ? "yes" : "no");

    printf("\t\tL2 cache size: %d bytes\n", dev_prop->l2CacheSize);
    printf("\t\tConstant memory size: %lu Mbytes\n", dev_prop->totalConstMem / (1024*1024));
    printf("\t\tShared memory size per block: %lu bytes\n",
                 dev_prop->sharedMemPerBlock);

    printf("\t\tNumber of registers available per block: %d\n",
                 dev_prop->regsPerBlock);
    printf("\t\tMax number of threads per block: %d\n",
                 dev_prop->maxThreadsPerBlock);
    printf("\t\tNumber of registers available per thread: %d\n",
                 dev_prop->regsPerBlock / dev_prop->maxThreadsPerBlock);

    printf("\t\tWarp size: %d\n", dev_prop->warpSize);
    printf("\t\tMax number of threads per multiprocessor: %d\n",
                 dev_prop->maxThreadsPerMultiProcessor);
    printf("\t\tNumber of multiprocessors: %d\n", dev_prop->multiProcessorCount);

    printf("\t\tMax sizes of each dimension of a block: (%d x %d x %d)\n",
                 dev_prop->maxThreadsDim[0],
                 dev_prop->maxThreadsDim[1],
                 dev_prop->maxThreadsDim[2]);
    printf("\t\tMax sizes of each dimension of a grid: (%d x %d x %d)\n",
                 dev_prop->maxGridSize[0],
                 dev_prop->maxGridSize[1],
                 dev_prop->maxGridSize[2]);
    printf("\t\tConcurrent copy and execution? %s\n",
                 dev_prop->deviceOverlap ? "yes" : "no");
    printf("\t\tLaunch concurrent kernels? %s\n",
                 dev_prop->concurrentKernels ? "yes" : "no");
    //printf("\t\tSingle/Double precision performance ratio: %d\n",
    //             dev_prop->singleToDoublePrecisionPerfRatio);
    printf("\t\tNumber of asynchronous engines: %d\n",
                 dev_prop->asyncEngineCount);
    //printf("\t\tNative atomic operations between host and device?: %s\n",
    //             dev_prop->hostNativeAtomicSupported ? "yes" : "no");
    
    printf("[+] end CUDA info\n");
}
}
