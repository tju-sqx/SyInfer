#ifndef SCALE_SUM_KERNEL_CPU_H
#define SCALE_SUM_KERNEL_CPU_H

#include <armadillo>
#include "tensor.h"

namespace kernel {
    void scale_sum_kernel(const Tensor& score_tensor, const Tensor& value_tensor, const Tensor& output_tensor, size_t head_size, size_t pos, size_t step);
}
#endif
// SCALE_SUM_KERNEL_CPU_H