## Compile and Run

```bash
$ nvcc --cudart shared get_device_info.c cuda_util.cu -o get_device_info
```

```bash
$ ./get_device_info
```

## Sample Output

```
[+] CUDA info:
                Runtime version: 9.1
                Capability: 6.1
                Global memory size: 2048 Mbytes
                GPU clock rate: 1417.00 MHz
                Memory clock rate: 0.00 MHz
                Memory bus width: 0 bits
                Max memory pitch: 0 bytes
                L1 support global cache? no
                L1 support local cache? no
                L2 cache size: 0 bytes
                Constant memory size: 1024 Mbytes
                Shared memory size per block: 49152 bytes
                Number of registers available per block: 32768
                Max number of threads per block: 1024
                Number of registers available per thread: 32
                Warp size: 32
                Max number of threads per multiprocessor: 1024
                Number of multiprocessors: 56
                Max sizes of each dimension of a block: (1024 x 1024 x 64)
                Max sizes of each dimension of a grid: (1073741824 x 1073741824 x 1073741824)
                Concurrent copy and execution? no
                Launch concurrent kernels? no
                Number of asynchronous engines: 0
[+] end CUDA info
```

