#ifndef MATMUL_KERNEL_CPU_H
#define MATMUL_KERNEL_CPU_H

#include <armadillo>
#include "tensor.h"

namespace kernel {
    void matmul_kernel_cpu(const Tensor& input_tensor, const Tensor& weight_tensor, const Tensor& output_tensor, float scale = 1.f);
}
#endif
//MATMUL_KERNEL_CPU_H