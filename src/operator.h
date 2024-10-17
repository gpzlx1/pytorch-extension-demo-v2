#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/script.h>

torch::Tensor AddCUDA(torch::Tensor a, torch::Tensor b, torch::Tensor c);