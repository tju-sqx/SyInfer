#ifndef SOFTMAX_KERNEL_CPU_H
#define SOFTMAX_KERNEL_CPU_H

#include <armadillo>
#include "tensor.h"

namespace kernel{
    void softmax_kernel_cpu(const Tensor& t_input, void* ptr);
}

#endif
//SOFTMAX_KERNEL_CPU_H