#ifndef RMSNORM_KERNEL_CPU_H
#define RMSNORM_KERNEL_CPU_H

#include <armadillo>
#include "tensor.h"

namespace kernel{
    void rmsnorm_kernel_cpu(const Tensor& input_tensor, const Tensor& weight_tensor, Tensor& output_tensor, void* stream);
}
#endif
// RMSNORM_KERNEL_CPU_H