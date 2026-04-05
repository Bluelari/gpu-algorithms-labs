#pragma once

#include "shape.hpp"

enum class ConvAlgorithm {
  Baseline = 0,
};

void convlayer_gpu_opt(const float *X, const shape &xdims, const float *W, const shape &wdims, float *Y, const shape &ydims,
                       ConvAlgorithm algorithm);
