#pragma once

#include "shape.hpp"

enum class ConvAlgorithm {
  CuDNN = -1,
  Baseline = 0,
  MatmulConceptualUnrollingRegisterTiled = 1,
};

void convlayer_gpu_opt(const float *X, const shape &xdims, const float *W, const shape &wdims, float *Y, const shape &ydims,
                       ConvAlgorithm algorithm);
