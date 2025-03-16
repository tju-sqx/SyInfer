#ifndef EMB_KERNEL_CPU_H
#define EMB_KERNEL_CPU_H

#include <armadillo>
#include "tensor.h"

namespace kernel{
    void emb_kernel_cpu(const Tensor& t_input, const Tensor& t_weight, Tensor& t_output, void* ptr);
}

#endif
//EMB_KERNEL_CPU_H END