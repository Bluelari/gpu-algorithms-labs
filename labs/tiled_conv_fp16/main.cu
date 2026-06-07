#include <cuda_fp16.h>
#include <iostream>

#include "helper.hpp"
#include "kernel.hpp"
#include "shape.hpp"

// Baseline GPU kernel code for forward convolution.
// One thread per output index
// You should not modify this kernel as it is used for correctness comparison.
// Instead, define a new one below
__global__ void conv_forward_baseline_kernel(const __half *X, const shape xdims, const __half *W, const shape wdims, __half *Y,
                                             const shape ydims) {

  const size_t gx = blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t i = gx; i < ydims.num * ydims.depth * ydims.height * ydims.width; i += blockDim.x * gridDim.x) {
    Y[i] = 0.f;
  }

  for (size_t i = gx; i < ydims.num; i += gridDim.x * blockDim.x) {
    for (auto m : range(0, ydims.depth)) {    // for each output feature map
      for (auto h : range(0, ydims.height)) { // for each output element
        for (auto w : range(0, ydims.width)) {
          const size_t yoffset = ((i * ydims.depth + m) * ydims.height + h) * ydims.width + w;
          for (auto c : range(0, xdims.depth)) {     // sum over all input feature maps
            for (auto p : range(0, wdims.height)) {  // filter height
              for (auto q : range(0, wdims.width)) { // filter width
                const size_t xoffset = ((((i * xdims.depth) + c) * xdims.height) + (h + p)) * xdims.width + (w + q);
                const size_t woffset = ((((m * wdims.depth) + c) * wdims.height) + p) * wdims.width + q;
                Y[yoffset] += X[xoffset] * W[woffset];
              }
            }
          }
        }
      }
    }
  }
}

// Host code to configure baseline GPU kernel
static void convlayer_gpu_baseline(const __half *X, const shape &xdims, const __half *W, const shape &wdims, __half *Y, const shape &ydims) {

  dim3 dimGrid(1);
  dim3 dimBlock(32);

  conv_forward_baseline_kernel<<<dimGrid, dimBlock>>>(X, xdims, W, wdims, Y, ydims);
  THROW_IF_ERROR(cudaGetLastError());
}

static int eval(const shape wDims, const shape xDims, bool doVerify, ConvAlgorithm algorithm) {

  // Generate model
  const auto conf_info = std::string("conv[wDims:") + std::to_string(wDims.num) + "," + std::to_string(wDims.depth) + "," +
                         std::to_string(wDims.height) + "," + std::to_string(wDims.width) + " xDims:" + std::to_string(xDims.num) + "," +
                         std::to_string(xDims.depth) + "," + std::to_string(xDims.height) + "," + std::to_string(xDims.width) + "]";
  INFO("Running " << conf_info);

  // Generate convolution weights
  float *hostW = allocate<float>(wDims);
  generate_convfilters(hostW, wDims);

  // generate input feature map
  float *hostX = allocate<float>(xDims);
  generate_data(hostX, xDims);

  // generate output feature map for verification
  const shape ydims = {xDims.num, wDims.num, (xDims.height - wDims.height + 1), (xDims.width - wDims.width + 1)};
  INFO("Allocating output tensor [" << ydims.num << "," << ydims.depth << "," << ydims.height << "," << ydims.width << "]");
  float *hostY    = allocate<float>(ydims);
  float *expected = allocate<float>(ydims);
  generate_data(hostY, ydims);

  // Convert float arrys to __half.
  __half *hostW_half = allocate<__half>(wDims);
  __half *hostX_half = allocate<__half>(xDims);
  __half *hostY_half = allocate<__half>(ydims);
  __half *expected_half = allocate<__half>(ydims);

  for (int i = 0; i < wDims.flattened_length(); i++) {
    hostW_half[i] = __float2half(hostW[i]);
  }
  for (int i = 0; i < xDims.flattened_length(); i++) {
    hostX_half[i] = __float2half(hostX[i]);
  }
  for (int i = 0; i < ydims.flattened_length(); i++) {
    hostY_half[i] = __float2half(hostY[i]);
  }


  const size_t wByteCount = wDims.flattened_length() * sizeof(__half);
  const size_t xByteCount = xDims.flattened_length() * sizeof(__half);
  const size_t yByteCount = ydims.flattened_length() * sizeof(__half);

  __half *deviceW = nullptr, *deviceX = nullptr, *deviceY = nullptr;
  timer_start("Allocating GPU memory.");
  THROW_IF_ERROR(cudaMalloc((void **) &deviceW, wByteCount));
  THROW_IF_ERROR(cudaMalloc((void **) &deviceX, xByteCount));
  THROW_IF_ERROR(cudaMalloc((void **) &deviceY, yByteCount));
  timer_stop();

  timer_start("Copying inputs to the GPU.");
  THROW_IF_ERROR(cudaMemcpy(deviceW, hostW_half, wByteCount, cudaMemcpyDefault));
  THROW_IF_ERROR(cudaMemcpy(deviceX, hostX_half, xByteCount, cudaMemcpyDefault));
  timer_stop();

  //////////////////////////////////////////
  // GPU Gather Computation
  //////////////////////////////////////////
  timer_start("Performing GPU convlayer");
  convlayer_gpu_opt(deviceX, xDims, deviceW, wDims, deviceY, ydims, algorithm);
  THROW_IF_ERROR(cudaDeviceSynchronize());
  timer_stop();

  // verify with provided implementation
  if (doVerify) {
    timer_start("Copying output to the CPU");
    THROW_IF_ERROR(cudaMemcpy(hostY_half, deviceY, yByteCount, cudaMemcpyDefault));
    for (int i = 0; i < ydims.flattened_length(); i++) {
      hostY[i] = __half2float(hostY_half[i]);
    }
    timer_stop();

    convlayer_gpu_baseline(deviceX, xDims, deviceW, wDims, deviceY, ydims);
    THROW_IF_ERROR(cudaDeviceSynchronize());
    THROW_IF_ERROR(cudaMemcpy(expected_half, deviceY, yByteCount, cudaMemcpyDefault));
    for (int i = 0; i < ydims.flattened_length(); i++) {
      expected[i] = __half2float(expected_half[i]);
    }

    verify(expected, hostY, ydims);
  }

  THROW_IF_ERROR(cudaFree(deviceW));
  THROW_IF_ERROR(cudaFree(deviceX));
  THROW_IF_ERROR(cudaFree(deviceY));
  free(hostW);
  free(hostX);
  free(hostY);
  free(expected);
  free(hostW_half);
  free(hostX_half);
  free(hostY_half);
  free(expected_half);

  return 0;
}

TEST_CASE("Convlayer", "[convlayer]") {
  ConvAlgorithm algorithm = ConvAlgorithm::CuDNN;
#if 1
  SECTION("[wDims:16,1,5,5 xDims:20,1,28,28]") {
    eval({16, 1, 5, 5}, {20, 1, 28, 28}, true, algorithm);
  }

  // // Test channel != 1
  // SECTION("[wDims:32,3,5,5 xDims:20,3,28,28]") {
  //   eval({32, 3, 5, 5}, {20, 3, 28, 28}, true, algorithm);
  // }
  // SECTION("[wDims:2,15,5,5 xDims:20,15,28,28]") {
  //   eval({32, 15, 5, 5}, {20, 15, 28, 28}, true, algorithm);
  // }

  // // Test filter != 32
  // SECTION("[wDims:4,1,5,5 xDims:20,1,28,28]") {
  //   eval({4, 1, 5, 5}, {20, 1, 28, 28}, true, algorithm);
  // }
  // SECTION("[wDims:8,1,5,5 xDims:20,1,28,28]") {
  //   eval({8, 1, 5, 5}, {20, 1, 28, 28}, true, algorithm);
  // }
  // SECTION("[wDims:16,1,5,5 xDims:20,1,28,28]") {
  //   eval({16, 1, 5, 5}, {20, 1, 28, 28}, true, algorithm);
  // }
  // SECTION("[wDims:5,1,5,5 xDims:20,1,28,28]") {
  //   eval({5, 1, 5, 5}, {20, 1, 28, 28}, true, algorithm);
  // }
  // SECTION("[wDims:18,1,5,5 xDims:20,1,28,28]") {
  //   eval({18, 1, 5, 5}, {20, 1, 28, 28}, true, algorithm);
  // }
  // SECTION("[wDims:1,1,5,5 xDims:20,1,28,28]") {
  //   eval({1, 1, 5, 5}, {20, 1, 28, 28}, true, algorithm);
  // }

  // // Test input size != 28 x 28
  // SECTION("[wDims:3,2,5,5 xDims:10,2,56,56]") {
  //   eval({3, 2, 5, 5}, {10, 2, 56, 56}, true, algorithm);
  // }
  // SECTION("[wDims:3,2,5,5 xDims:10,2,112,112]") {
  //   eval({3, 2, 5, 5}, {10, 2, 112, 112}, true, algorithm);
  // }
  // SECTION("[wDims:3,2,5,5 xDims:10,2,20,28]") {
  //   eval({3, 2, 5, 5}, {10, 2, 20, 28}, true, algorithm);
  // }
  // SECTION("[wDims:3,2,5,5 xDims:10,2,28,20]") {
  //   eval({3, 2, 5, 5}, {10, 2, 28, 20}, true, algorithm);
  // }
  // SECTION("[wDims:3,2,5,5 xDims:10,2,6,8]") {
  //   eval({3, 2, 5, 5}, {10, 2, 6, 8}, true, algorithm);
  // }
  // SECTION("[wDims:3,2,5,5 xDims:10,2,31,31]") {
  //   eval({3, 2, 5, 5}, {10, 2, 31, 31}, true, algorithm);
  // }
  // SECTION("[wDims:3,2,5,5 xDims:10,2,7,7]") {
  //   eval({3, 2, 5, 5}, {10, 2, 7, 7}, true, algorithm);
  // }
  // SECTION("[wDims:3,2,5,5 xDims:10,2,25,28]") {
  //   eval({3, 2, 5, 5}, {10, 2, 25, 28}, true, algorithm);
  // }
  // SECTION("[wDims:3,2,5,5 xDims:10,2,28,25]") {
  //   eval({3, 2, 5, 5}, {10, 2, 28, 25}, true, algorithm);
  // }
  // SECTION("[wDims:3,2,5,5 xDims:10,2,5,5]") {
  //   eval({3, 2, 5, 5}, {10, 2, 5, 5}, true, algorithm);
  // }

  // // Test kernel size = 3x3
  // SECTION("[wDims:3,1,3,3 xDims:10,1,28,28]") {
  //   eval({3, 1, 3, 3}, {10, 1, 28, 28}, true, algorithm);
  // }
  // SECTION("[wDims:3,2,3,3 xDims:10,2,28,28]") {
  //   eval({3, 2, 3, 3}, {10, 2, 28, 28}, true, algorithm);
  // }
  // SECTION("[wDims:3,1,3,3 xDims:10,1,56,56]") {
  //   eval({3, 1, 3, 3}, {10, 1, 56, 56}, true, algorithm);
  // }
  // SECTION("[wDims:3,2,3,3 xDims:10,2,56,56]") {
  //   eval({3, 2, 3, 3}, {10, 2, 56, 56}, true, algorithm);
  // }
  // SECTION("[wDims:3,2,3,3 xDims:10,2,3,3]") {
  //   eval({3, 2, 3, 3}, {10, 2, 3, 3}, true, algorithm);
  // }
  // SECTION("[wDims:3,2,3,3 xDims:10,2,28,25]") {
  //   eval({3, 2, 3, 3}, {10, 2, 28, 25}, true, algorithm);
  // }
  // SECTION("[wDims:3,2,3,3 xDims:10,2,25,28]") {
  //   eval({3, 2, 3, 3}, {10, 2, 25, 28}, true, algorithm);
  // }
#else
  SECTION("[wDims:16,3,5,5 xDims:5000,3,28,28]") {
    eval({16, 1, 5, 5}, {5000, 1, 28, 28}, false, algorithm);
  }
  SECTION("[wDims:16,3,5,5 xDims:5000,3,56,56]") {
    eval({16, 1, 5, 5}, {5000, 1, 56, 56}, false, algorithm);
  }
  SECTION("[wDims:16,3,5,5 xDims:5000,3,112,112]") {
    eval({16, 1, 5, 5}, {5000, 1, 112, 112}, false, algorithm);
  }
  SECTION("[wDims:16,3,5,5 xDims:5000,3,28,28]") {
    eval({16, 3, 5, 5}, {5000, 3, 28, 28}, false, algorithm);
  }
  SECTION("[wDims:16,3,5,5 xDims:5000,3,56,56]") {
    eval({16, 3, 5, 5}, {5000, 3, 56, 56}, false, algorithm);
  }
  SECTION("[wDims:16,3,5,5 xDims:5000,3,112,112]") {
    eval({16, 3, 5, 5}, {5000, 3, 112, 112}, false, algorithm);
  }
  SECTION("[wDims:16,6,5,5 xDims:5000,6,28,28]") {
    eval({16, 6, 5, 5}, {5000, 6, 28, 28}, false, algorithm);
  }
  SECTION("[wDims:16,6,5,5 xDims:5000,6,56,56]") {
    eval({16, 6, 5, 5}, {5000, 6, 56, 56}, false, algorithm);
  }
  SECTION("[wDims:16,6,5,5 xDims:5000,6,112,112]") {
    eval({16, 6, 5, 5}, {5000, 6, 112, 112}, false, algorithm);
  }

  SECTION("[wDims:16,3,3,3 xDims:5000,3,28,28]") {
    eval({16, 1, 3, 3}, {5000, 1, 28, 28}, false, algorithm);
  }
  SECTION("[wDims:16,3,3,3 xDims:5000,3,56,56]") {
    eval({16, 1, 3, 3}, {5000, 1, 56, 56}, false, algorithm);
  }
  SECTION("[wDims:16,3,3,3 xDims:5000,3,112,112]") {
    eval({16, 1, 3, 3}, {5000, 1, 112, 112}, false, algorithm);
  }
  SECTION("[wDims:16,3,3,3 xDims:5000,3,28,28]") {
    eval({16, 3, 3, 3}, {5000, 3, 28, 28}, false, algorithm);
  }
  SECTION("[wDims:16,3,3,3 xDims:5000,3,56,56]") {
    eval({16, 3, 3, 3}, {5000, 3, 56, 56}, false, algorithm);
  }
  SECTION("[wDims:16,3,3,3 xDims:5000,3,112,112]") {
    eval({16, 3, 3, 3}, {5000, 3, 112, 112}, false, algorithm);
  }
  SECTION("[wDims:16,6,3,3 xDims:5000,6,28,28]") {
    eval({16, 6, 3, 3}, {5000, 6, 28, 28}, false, algorithm);
  }
  SECTION("[wDims:16,6,3,3 xDims:5000,6,56,56]") {
    eval({16, 6, 3, 3}, {5000, 6, 56, 56}, false, algorithm);
  }
  SECTION("[wDims:16,6,3,3 xDims:5000,6,112,112]") {
    eval({16, 6, 3, 3}, {5000, 6, 112, 112}, false, algorithm);
  }
#endif
}
