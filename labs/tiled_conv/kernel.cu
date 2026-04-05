#include "kernel.hpp"

#include <cstddef>
#include <cuda_runtime.h>
#include <iostream>

#include "common/fmt.hpp"
#include "common/utils.hpp"

// Computes ceil(x / y)
std::size_t ceil_div(std::size_t x, std::size_t y) {
  std::size_t offset = x % y == 0 ? 0 : 1;
  return x / y + offset;
}

// A better baseline than the one defined in main.cu.
// 1 thread outputs 1 feature image.
__global__ void baseline_conv_kernel(const float *X, const shape xdims, const float *W, const shape wdims, float *Y, const shape ydims) {
  const std::size_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (global_idx < ydims.num * ydims.depth) {
    std::size_t batch_idx  = global_idx / ydims.depth; // this thread process this input image
    std::size_t filter_idx = global_idx % ydims.depth; // and this filter

    for (std::size_t i = 0; i < ydims.height; i++) {
      for (std::size_t j = 0; j < ydims.width; j++) {
        // Compute Y[batch_idx][filter_idx][i][j]
        float output = 0.0f;
        for (std::size_t c = 0; c < wdims.depth; c++) {
          for (std::size_t p = 0; p < wdims.height; p++) {
            for (std::size_t q = 0; q < wdims.width; q++) {
              // Y[batch_idx][filter_idx][i][j] += X[batch_idx][c][i+p][j+q] * W[filter_idx][c][p][q]
              float x = X[((batch_idx * xdims.depth + c) * xdims.height + (i + p)) * xdims.width + (j + q)];
              float w = W[((filter_idx * wdims.depth + c) * wdims.height + p) * wdims.width + q];
              output += x * w;
            }
          }
        }
        Y[((batch_idx * ydims.depth + filter_idx) * ydims.height + i) * ydims.width + j] = output;
      }
    }
  }
}

void convlayer_gpu_opt(const float *X, const shape &xdims, const float *W, const shape &wdims, float *Y, const shape &ydims,
                       ConvAlgorithm algorithm) {
  switch (algorithm) {
    case ConvAlgorithm::Baseline: {
      dim3 dim_grid(ceil_div(ydims.num * ydims.depth, 256));
      dim3 dim_block(256); // This size is chosen arbitrarily.
      baseline_conv_kernel<<<dim_grid, dim_block>>>(X, xdims, W, wdims, Y, ydims);
      THROW_IF_ERROR(cudaGetLastError());
      THROW_IF_ERROR(cudaDeviceSynchronize());
      break;
    }
    default:
      std::cerr << "Invalid convolution algorithm {}" << static_cast<int>(algorithm) << std::endl;
  }
}
