#include "mha_kernel.h"
#include "alloc.h"
#include "alloc_cpu.h"
#include "softmax_kernel_cpu.h"
#include "matmul_kernel_cpu.h"
#include "scale_sum_kernel.h"
#include <cstdio>
#include <glog/logging.h>

namespace kernel {
    void mha_kernel(size_t pos, size_t seq_len, size_t kv_dim, size_t head_num, size_t head_size, size_t kv_mul, size_t layer_idx,
    const Tensor& mha_out, const Tensor& query_tensor, const Tensor& key_tensor, const Tensor& value_tensor, const Tensor& score_tensor, base::DeviceType device_type) {
        size_t layer_offset = layer_idx * seq_len * kv_dim;
        float scale = 1.0f / std::sqrt(static_cast<float>(head_size));

        std::shared_ptr<DeviceAlloc> allocator = nullptr;
        if(device_type == base::DeviceType::CPU) {
            allocator = CpuAllocFactory::get_instance();
        }
        for (size_t h = 0; h < head_num; ++h) {
            float* score_ptr = const_cast<float*>(score_tensor.data<float>()) + h * seq_len;
            float* query_ptr = const_cast<float*>(query_tensor.data<float>() + h * head_size);

            for (size_t p = 0; p <= pos; ++p) {
                size_t key_offset = p * kv_dim + (h / kv_mul) * head_size;
                float* key_ptr = const_cast<float*>(key_tensor.data<float>()) + layer_offset + key_offset;

                Tensor query_vec = Tensor{head_size, base::DateType::DATA_FP32, nullptr, query_ptr};
                Tensor key_mat = Tensor{1, head_size, base::DateType::DATA_FP32, nullptr, key_ptr};

                Tensor score_element = Tensor{1, base::DateType::DATA_FP32, nullptr, score_ptr + p};
                query_vec.set_device_type(device_type);
                key_mat.set_device_type(device_type);
                score_element.set_device_type(device_type);

                query_vec.create();
                key_mat.create();
                score_element.create();

                (void)matmul_kernel_cpu(query_vec, key_mat, score_element, scale);
            }

            Tensor score_vec{pos + 1, base::DateType::DATA_FP32, nullptr, score_ptr};
            score_vec.create();
            score_vec.set_device_type(device_type);  
            softmax_kernel_cpu(score_vec, nullptr);
            float* mha_out_ptr = const_cast<float*>(mha_out.data<float>()) + h * head_size;
            allocator->memset_zero(mha_out_ptr, sizeof(float) * head_size);
            Tensor output_vec = Tensor{head_size, base::DateType::DATA_FP32, nullptr, mha_out_ptr};
            output_vec.create();
            output_vec.set_device_type(device_type);

            size_t v_offset = head_size * (h / kv_mul);
            //std::cout<<"layer_offset "<<layer_offset<<"value_offset "<<v_offset<<"\n";
            float* value_ptr = const_cast<float*>(value_tensor.data<float>()) + v_offset + layer_offset;
            for(int i = v_offset; i < head_size + v_offset; i++) {
                //std::cout<<"idx: "<<i<<" value element: "<<*(value_ptr + i)<<"\n";
            }
            Tensor value_vec = Tensor{head_size, base::DateType::DATA_FP32, nullptr, value_ptr};
            value_vec.create();
            value_vec.set_device_type(device_type);
            //std::cout<<"value_vec size "<<value_vec.size()<<"\n";
            (void)scale_sum_kernel(score_vec, value_vec, output_vec, head_size, pos, kv_dim);
        }

    }
}