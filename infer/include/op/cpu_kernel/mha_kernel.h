#ifndef MHA_KERNEL_H
#define MHA_KERNEL_H

#include <armadillo>
#include "tensor.h"

namespace kernel{
    void mha_kernel(size_t pos, size_t seq_len, size_t kv_dim, size_t head_num, size_t head_size, size_t kv_mul, size_t layer_idx,
    const Tensor& mha_out, const Tensor& query_tensor, const Tensor& key_tensor, const Tensor& value_tensor, const Tensor& score_tensor, base::DeviceType device_type);
}

#endif