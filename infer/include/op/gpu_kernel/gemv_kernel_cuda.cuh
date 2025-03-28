#ifndef GEMV_KERNEL_CU_CUH
#define GEMV_KERNEL_CU_CUH
#include "base.h"
#include "tensor.h"
#include <cstddef>
#include <glog/logging.h>

namespace kernel {
    void gemv_kernel_cuda(const Tensor& input_tensor, const Tensor& weight_tensor, const Tensor& output_tensor, float scale = 1.0f);
}
#endif