#include "rope_kernel_cpu.h"
#include <glog/logging.h>
#include <cmath>

namespace kernel{
    void sin_cos_calc(size_t head_size, size_t max_len_size, float* sin_cache, float* cos_cache) {
        for(size_t pos = 0; pos < max_len_size; ++pos) {
            for(size_t head_dim = 0; head_dim < head_size; ++head_dim) {
                float frec = (1.0f / pow(10000.0f, static_cast<float>(head_dim) / static_cast<float>(head_size)));
                float rotate = static_cast<float>(pos) * frec;
                *(sin_cache + pos * head_size + head_dim) = sinf(rotate);
                *(cos_cache + pos * head_size + head_dim) = cosf(rotate);
            }      
        }
    }

    void rope_kernel_cpu(size_t dim_size, size_t k_size, size_t head_size, Tensor& pos_input, Tensor& k_input, Tensor& q_input, Tensor& sin_cache, Tensor& cos_cache) {
        size_t pos = static_cast<float>(*pos_input.data<size_t>());

        for(size_t dim = 0; dim < dim_size; dim += 2) {
            size_t head_dim = dim % head_size;
            //std::cout<<"dim:" <<dim<<"\n";
            float sin_value = static_cast<float>(*((sin_cache.data<float>()) + pos * head_size + head_dim));
            float cos_value = static_cast<float>(*((cos_cache.data<float>()) + pos * head_size + head_dim));

            size_t qk_size = dim < k_size? 2 : 1;
            for(size_t i = 0; i < qk_size; ++i) {
                float* vec = i == 0? (q_input.data<float>()) : (k_input.data<float>());

                float vec_dim_i = vec[dim];
                float vec_dim_i_plus_one = vec[dim + 1];
                // std::cout<<"vec_dim_i:" <<dim<<" "<< vec_dim_i<<" sin "<<sin_value<<"\n";
                // std::cout<<"vec_dim_i_plus_ont:" <<dim + 1<<" "<< vec_dim_i_plus_one<<" cos "<<cos_value<<"\n";
                vec[dim] = vec_dim_i * cos_value - vec_dim_i_plus_one * sin_value;
                vec[dim + 1] = vec_dim_i * sin_value + vec_dim_i_plus_one * cos_value;
            }
        }
    }
}