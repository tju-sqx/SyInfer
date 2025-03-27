#include "emb_kernel_cpu.h"
#include <glog/logging.h>

namespace kernel {
    void emb_kernel_cpu(const Tensor& t_input, const Tensor& t_weight, Tensor& t_output, void* ptr) {
        CHECK(t_input.device_type() == base::DeviceType::CPU) << "emb_kernel cpu only accept CPU data \n";
        CHECK(t_input.device_type() == t_weight.device_type() 
        && t_input.device_type() == t_output.device_type()) << "emb_kernel_cpu tensor has wrong device type \n";
        CHECK(!t_input.empty()) << "emb_kernel_cpu t_input is empty \n";
        CHECK(!t_weight.empty()) << "emb_kernel_cpu t_weight is empty \n";

        //TODO: weight size and output size
        const size_t t_input_size = t_input.size();
        const size_t weight_dim = t_weight.get_dim(1);

        const float* input_data = t_input.data<float>();
        for (size_t i = 0; i < t_input_size; ++i) {
            const int32_t tokenid = input_data[i];
            //TODO: vocab size
            const float* weight_data = t_weight.data<float>();
            float* output_data = t_output.data<float>();

            memcpy(output_data + i * weight_dim, weight_data + tokenid * weight_dim, sizeof(float) * weight_dim);
        }
    }
}