#include "scale_sum_kernel.h"
#include <glog/logging.h>
#include <iostream>


namespace kernel {
    void scale_sum_kernel(const Tensor& score_tensor, const Tensor& value_tensor, const Tensor& output_tensor, size_t head_size, size_t pos, size_t step) {
        CHECK(score_tensor.device_type() == value_tensor.device_type());
        CHECK(score_tensor.device_type() == output_tensor.device_type());
        CHECK(!score_tensor.empty());
        CHECK(!value_tensor.empty());
        CHECK(!output_tensor.empty());
        CHECK(score_tensor.size() == pos + 1) << "score_tensor has wrong size in scale_sum_kernel func, tensor size is " <<score_tensor.size() <<"\n";
        CHECK(value_tensor.size() == head_size) << "value_tensor has wrong size in scale_sum_kernel func, tensor size is " <<value_tensor.size() <<"\n";
        CHECK(output_tensor.size() == head_size) << "output_tensor has wrong size in scale_sum_kernel func, tensor size is " <<output_tensor.size() <<"\n";

        float* score_ptr = const_cast<float*>(score_tensor.data<float>());
        float* output_ptr = const_cast<float*>(output_tensor.data<float>());
        
        arma::fvec score_vec(score_ptr, pos + 1, false, true);
        arma::fvec output_vec(output_ptr, head_size, false, true);
        for(size_t p = 0; p <= pos; ++p) {
            arma::fvec value_vec(const_cast<float*>(value_tensor.data<float>()) + p * step, head_size, false, true);
            output_vec += score_vec[p] * value_vec;
        }

    }
}