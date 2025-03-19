#ifndef ROPE_KERNEL_CPU_H
#define ROPE_KERNEL_CPU_H
#include "tensor.h"
#include <armadillo>
namespace kernel{
    void sin_cos_calc(size_t head_size, size_t max_len_size, float* sin_cache, float* cos_cache);
    void rope_kernel_cpu(size_t dim_size, size_t k_dim, size_t head_size, Tensor& pos_input, Tensor& k_input, Tensor& q_input, Tensor& sin_cache, Tensor& cos_cache); 
}
#endif
// ROPE_KERNEL_CPU_H END