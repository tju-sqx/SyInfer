#ifndef SWIGLU_KERNEL_CPU_H
#define SWIGLU_KERNEL_CPU_H

#include <armadillo>
#include "tensor.h"

namespace kernel{
    void swiglu_kernel_cpu(const Tensor& t_input, const Tensor& t_weight, Tensor& t_output, void* ptr);
}

#endif
//SWIGLU_KERNEL_CPU_H END