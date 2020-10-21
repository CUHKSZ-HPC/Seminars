## Run CUDA Codes On GPU Emulator

### Step 1: Get Docker Runtime

Perhaps you don't really have a NVIDIA GPU standby, or you haven't setup a CUDA environment, but it does not mean that you could not try CUDA programming on your own computer. I found there's someone on the Internet has already implemented a GPU emulator over CPU and he also published a docker image with GPU emulation enabled, which allows you to compile and execute CUDA codes even without a GPU.

Before getting start, you should first install a docker runtime on your host machine (no OS restriction). To install docker runtime, please follow this website: https://docs.docker.com/get-docker/



### Step 2: Pull, Start and Run the Docker

After installed Docker runtime, you are ready to download the docker image of GPU emulator and launch a docker container using the downloaded docker image.

In your Terminal (PowerShell for Windows Users) use following command to pull the docker image

```bash
$ docker pull srirajpaul/gpgpu-sim:0.1
```

Then, launch a docker container using the docker image

```bash
$ docker run -w /root -it srirajpaul/gpgpu-sim:0.1 /bin/bash
```

If everything goes right, the terminal will be connected to the docker container under the root directory

```bash
root@fd9611dbae32:~#
```

If  you want to exit the docker container, use `ctrl+p` and `ctrl+q` to detach the docker container. 

Next time when you want to re-attach the docker container, you could first obtain the docker id using `docker container ls` and you would get a similar output as follow

```
CONTAINER ID        IMAGE                      COMMAND             CREATED             STATUS              PORTS               NAMES
fd9611dbae32        srirajpaul/gpgpu-sim:0.1   "/bin/bash"         2 hours ago         Up 2 hours                              jolly_rhodes
ce205fd3bd23        vitowu/idash:latest        "/bin/bash"         15 hours ago        Up 15 hours                             naughty_cerf
81d779bb756c        vitowu/idash:latest        "/bin/bash"         15 hours ago        Up 15 hours                             serene_curran
```

From the output you could see that the docker id of image `srirajpaul/gpgpu-sim:0.1` is `fd9611dbae32`, so you could use following command to enter the docker container

```bash
$ docker container attach fd9611dbae32
```



### Step 3: Compile and Execute CUDA Program

Normally a CUDA program need to be compiled by an executable `nvcc` (NVIDIA CUDA compiler). Clearly, compilation staging in itself does not help towards the goal of application compatibility with future GPU, so nvcc will only help to compile the CUDA program in PTX, an intermediate representation for JIT virtual compute architecture.

![https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/graphics/just-in-time-compilation.png](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/graphics/just-in-time-compilation.png)

When the program execution starts, CUDA Runtime will proceed the compilation and regenerate the real SM codes from ptx file. In our case (inside the GPU emulator container), the `nvcc` compiler is almost the same as the official `nvcc` that you can download from NVIDIA, but the CUDA Runtime is emulated, so that after compilation you code will be executable without a GPU.

Under `/root/test` directory of the docker container, there is a sample program `vec_add.cu`. You can use following command to compile `vec_add.cu` into binary executable:

```bash
$ nvcc --cudart shared vec_add.cu -o vec_add
```

and then execute the binary `vec_add`

```bash
$ ./vec_add
```

If everything goes well, you will get following output:

```bash
root@fd9611dbae32:~/test# ./vec_add


        *** GPGPU-Sim Simulator Version 4.0.0  [build gpgpu-sim_git-commit-fb35d501e6c4f7f3919b008f98c997b8021d83a3_modified_0] ***


GPGPU-Sim PTX: simulation mode 1 detail 0 (can change with PTX_SIM_MODE_FUNC environment variable:
               1=functional simulation only, 0=detailed performance simulator)
GPGPU-Sim PTX: overriding embedded ptx with ptx file (PTX_SIM_USE_PTX_FILE is set)
9.1
acec5b55e503891c0bc96ff0ba89ae8c  /root/test/vec_add
Extracting PTX file and ptxas options    1: vec_add.1.sm_30.ptx -arch=sm_30
GPGPU-Sim PTX: __cudaRegisterFunction _Z4add1lPlS_S_ : hostFun 0x0x56419edb2796, fat_cubin_handle = 1
GPGPU-Sim PTX: __cudaRegisterFunction _Z3addlPlS_S_ : hostFun 0x0x56419edb26a7, fat_cubin_handle = 1
Size 128

GPGPU-Sim PTX: cudaLaunch for 0x0x56419edb26a7 (mode=functional simulation) on stream 0
GPGPU-Sim PTX: pushing kernel '_Z3addlPlS_S_' to stream 0, gridDim= (2,1,1) blockDim = (2,1,1)
GPGPU-Sim: Performing Functional Simulation, executing kernel _Z3addlPlS_S_...
Destroy streams for kernel 1: size 0
GPGPU-Sim: Done functional simulation (852 instructions simulated).


gpgpu_simulation_time = 0 days, 0 hrs, 0 min, 1 sec (1 sec)
gpgpu_simulation_rate = 852 (inst/sec)


GPGPU-Sim PTX: cudaLaunch for 0x0x56419edb2796 (mode=functional simulation) on stream 0
GPGPU-Sim PTX: pushing kernel '_Z4add1lPlS_S_' to stream 0, gridDim= (2,1,1) blockDim = (2,1,1)
GPGPU-Sim: Performing Functional Simulation, executing kernel _Z4add1lPlS_S_...
Destroy streams for kernel 2: size 0
GPGPU-Sim: Done functional simulation (1704 instructions simulated).


gpgpu_simulation_time = 0 days, 0 hrs, 0 min, 1 sec (1 sec)
gpgpu_simulation_rate = 1704 (inst/sec)

Equals test 1

GPGPU-Sim: *** exit detected ***
```



### Next Step: Now it's time to run your code on a REAL GPU

From Step 1 to Step 3 we are using a GPU emulator to execute the CUDA program. It is of certain that the emulator has limitations. You could use this emulator to easily validate the correctness of your programs, but you can never get the actual performance. The reason is very simple, the emulator does not built from a physical hardware, so the execution and memory model of your GPU programs will be far different from the real one that we have introduced in the seminar.

In order to inspect the actual execution performance and effectively optimize the performance of your program, you will inevitably need a real GPU for your experiments. Fortunately, we have 10+ GPUs available for you to test your programs. If you have written a correct CUDA program and you need to test the actual performance, please contact Levi and let him help you to get your program execute on a real GPU device.

