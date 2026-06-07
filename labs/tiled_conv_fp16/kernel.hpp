#pragma once

#include <cuda_fp16.h>

#include "shape.hpp"

enum class ConvAlgorithm {
  CuDNN = -1,
};

void convlayer_gpu_opt(
    const __half *X, const shape &xdims,
    const __half *W, const shape &wdims,
    __half *Y, const shape &ydims,
    ConvAlgorithm algorithm
);
