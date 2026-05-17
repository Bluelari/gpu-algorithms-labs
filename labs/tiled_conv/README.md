## Introduction

We implemented a 2D convolution forward algorithm based on matrix multiplication and benchmarked its performance using the Nvidia Nsight Compute (ncu) profiler. Our implementation achieved a speedup of *1.18x to 2.21x* compared to cuDNN's implementation.

## Task

The goal is to implement a 2D convolution forward algorithm with no padding and unit stride. All inputs and outputs are single-precision floating-point values (`float`).

We assume the inputs are in NCHW layout. The input image `X` has shape `N x C x H x W` and the convolution filter `W` has shape `K x C x R x S`, where
* `N`: batch size
* `C`: number of input channels
* `K`: number of output channels
* `H, W`: height and width of the input images
* `R, S`: height and width of the filter

The output feature map Y has shape `N x K x (H - R + 1) x (W - S + 1)`.

## Algorithm

The algorithm is based on GEMM over the conceptually unrolled input matrices and applies the following optimizations:
1. Register-tiled matrix multiplication: Inputs and outputs are loaded into tiles of registers before computing. Loops are unrolled whenever possible.
2. Thread coarsening: Each thread computes `K` output elements.
3. Filters in constant memory: Filter weights are stored in constant memory to exploit fast, read-only cache for broadcasting reads.
4. Matrix unrolling at the register level: implicit `im2col` happens when loading from shared memory to registers, instead of loading from global memory to shared memory.
5. Coalesced global memory access: As a consequence of optimization 4, global memory loads are coalesced.
6. Vectorized loads: `float2` is used when loading from global memory to reduce MIO throttle.

The algorithm enum of our final implementation is `ConvAlgorithm::MatmulConceptualUnrollingShmemRegisterTiled`.

## Setup

We compare the performance of our implementation against cuDNN's `cudnnConvolutionForward` API under the following conditions:
* We fix `N = 5000` and `K = 16` for all benchmarks.
* We tested `C = 1, 3, 6`, `H x W = 28 x 28, 56 x 56, 112 x 112`, `R x S = 3 x 3, 5 x 5`.
* Benchmarks are run on an 8-core, 32 MB memory container with an Nvidia T4 GPU. The SM frequency is locked to 1.5 GHz.

For the cuDNN baseline, we first call `cudnnFindConvolutionForwardAlgorithm` to select the best algorithm for every input size, and then use that algorithm. We empirically found that under all testing setup, `CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PPRECOMP_GEMM` is always chosen as the best algorithm. `CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING` and `CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED` are sometimes reported as the best algorithm but the occurrence is inconsistent and infrequent, so we used `CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PPRECOMP_GEMM` uniformly.

We report the execution time of the primary compute kernel. For cuDNN, 2 kernels are launched: One for precomputing some indices, and the other for computation so we only time the latter. For our implementation, we time `conv_forward_shmem_register_tiled_matmul_kernel`. Data transfer overhead (host-to-device, device-to-host, and device-to-constant memory) is excluded from the measurements.

The input images `X` is filled with random floats in 0 to 1, while the filter `W` is initialized to all ones, following the original lab setup.

## Result

The table below compares the execution time for our kernel versus cuDNN. `N = 5000` and `K = 16` are fixed for all benchmarks.
| setup                                     | cuDNN    | our kernel | speedup |
|-------------------------------------------|----------|------------|---------|
| `C = 1, H x W = 28 x 28, R x S = 3 x 3`   | 1.02ms   | 0.86ms     | 1.18x   |
| `C = 3, H x W = 28 x 28, R x S = 3 x 3`   | 1.48ms   | 0.98ms     | 1.51x   |
| `C = 6, H x W = 28 x 28, R x S = 3 x 3`   | 2.32ms   | 1.60ms     | 1.45x   |
| `C = 1, H x W = 28 x 28, R x S = 5 x 5`   | 1.20ms   | 0.76ms     | 1.58x   |
| `C = 3, H x W = 28 x 28, R x S = 5 x 5`   | 2.44ms   | 1.18ms     | 2.06x   |
| `C = 6, H x W = 28 x 28, R x S = 5 x 5`   | 4.41ms   | 2.40ms     | 1.83x   |
| `C = 1, H x W = 56 x 56, R x S = 3 x 3`   | 4.34ms   | 3.67ms     | 1.18x   |
| `C = 3, H x W = 56 x 56, R x S = 3 x 3`   | 6.27ms   | 4.18ms     | 1.50x   |
| `C = 6, H x W = 56 x 56, R x S = 3 x 3`   | 9.88ms   | 6.12ms     | 1.61x   |
| `C = 1, H x W = 56 x 56, R x S = 5 x 5`   | 5.59ms   | 3.46ms     | 1.61x   |
| `C = 3, H x W = 56 x 56, R x S = 5 x 5`   | 11.31ms  | 5.76ms     | 1.96x   |
| `C = 6, H x W = 56 x 56, R x S = 5 x 5`   | 20.65ms  | 11.56ms    | 1.78x   |
| `C = 1, H x W = 112 x 112, R x S = 3 x 3` | 17.91ms  | 15.22ms    | 1.18x   |
| `C = 3, H x W = 112 x 112, R x S = 3 x 3` | 26.00ms  | 17.13ms    | 1.52x   |
| `C = 6, H x W = 112 x 112, R x S = 3 x 3` | 45.70ms  | 25.48ms    | 1.80x   |
| `C = 1, H x W = 112 x 112, R x S = 5 x 5` | 25.72ms  | 14.59ms    | 1.76x   |
| `C = 3, H x W = 112 x 112, R x S = 5 x 5` | 63.98ms  | 28.84ms    | 2.21x   |
| `C = 6, H x W = 112 x 112, R x S = 5 x 5` | 132.68ms | 62.76ms    | 2.11x   |

Our kernel is faster than cuDNN's implementation in all the test cases, and has achieved 1.18x - 2.21x speedup. The profiling reports can be found in [benchmark](benchmark) directory and historical reports can be found in [benchmark_50000_1_28_28](benchmark_50000_1_28_28).

## How to run

To run the benchmark, configure and build with cmake
```bash
cmake -S . -B build
cmake --build build
```
and run with
```bash
./build/tiled_conv
```
This runs all the benchmarks with configurations described in the Setup section using our implementation.


Below are the instructions of the original lab.

------

# Register-Tiled Neural Network Convolution Layer

## Objective
The goal of this lab is to implement the forward operation of a convolution layer using register-tiled matrix multiplication.

## CUDA Implementation

The skeleton code provides a basic CPU and GPU implementation of forward propagation for a convolutional layer.

Your CUDA implementation will be compared with the GPU version for the correctness at the end of each step for correctness and evaluated based on its achieved performance.

As mentioned in lecture, your code must work correctly in order to receive
credit.  You must implement register tiling to receive full functionality
credit.  And your code must perform reasonably well (various student
submissions as well as the instructor's version will be compared), as
some points will be awarded for performance.

You should implement your own register-tiled matrix-multiplication convolution written in CUDA.
Apply any optimization you think would bring benefit and feel free to modify any part of the code.
Once you have an optimized version, you may be interested in comparing your implementation with `cuBLAS` or `cuDNN`.

## Dataset Information

There is one set of convolution layer parameters assigned in this lab. We use random function to generate the input data images, hence, all the input test datasets between students and between each runs will be unique. Therefore, it is important to make sure that the output values match the results from the baseline GPU code. For computation simplicity, convolution filter weights are all set to 1.

* Input Dataset `N x C x H x W = N x 1 x 28 x 28`: all random variables
* Filter `M x C x K x K =  32 x 1 x 5 x 5`: all 1's

If you look at the `eval` calls at the bottom of `main.cu`, you will see
an `ifdef` to switch between debugging, which uses N=20, and performance
measurement, which uses N=50,000.  Be sure that your code passes the
tests before turning it in (do so AFTER you have optimized performance as
well, just in case!).  Note that the performance case turns off verification
for speed, so you will not know that your code does not work if you
run only the performance version.


## Instructions

In the provided source code, you will find functions named `conv_forward_valid` and `conv_forward_kernel`.
These functions implement the sequential forward path and the GPU forward path of the convolution layer.
You do not have to modify these functions, they are used during validation.
You should modify the `conv_forward_opt_kernel` to implement your register-tiled kernel.
Don't forget to modify the host code in `convlayer_gpu_opt`.

You have to implement the host code to call GPU kernels, the GPU kernel functions and any additional CUDA memory management.
Once you have finished with CUDA implementation, you will be using the function `verify` to verify your solution with the results from the sequential code.
You will check the output feature maps, Y (or out), after the forward propagation.

You may optimize as much as you would like based on the layer parameters.
Do NOT optimize based on the weights all being set to 1.

## Range For Loops

Throughout the serial code, we use the [`range.hpp`][rangehpp] to make the code easier to understand. Essentially,

```{.cpp}
for (const auto ii : range(0, N)) {
    do_stuff(ii);
}
```

Is equivalent to

```{.cpp}
for (const auto ii = 0; ii < N; ii++) {
    do_stuff(ii);
}
```

The use of range introduces some overhead and you might get better speed if you remove it's usage.
