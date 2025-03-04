#ifndef RMSNORM_KERNEL_CPU_H
#define RMSNORM_KERNEL_CPU_H

#include <armadillo>
#include "tensor.h"

namespace kernel{
    void rmsnorm_kernel_cpu(const Tensor& t_input, const Tensor& t_weight, Tensor& t_output, void* ptr);
}

#endif