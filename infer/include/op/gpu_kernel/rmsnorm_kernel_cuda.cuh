#ifndef RMSNORM_KERNEL_CUDA_CUH
#define RMSNORM_KERNEL_CUDA_CUH
#include "tensor.h"

namespace kernel{
    void rmsnorm_kernel_cuda(const Tensor& input_tensor, const Tensor& weight_tensor, const Tensor& output_tensor, void* stream);
}
#endif
// RMSNORM_KERNEL_CUDA_CUH