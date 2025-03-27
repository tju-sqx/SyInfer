#include "softmax_kernel_cpu.h"
#include <armadillo>
#include <glog/logging.h>

namespace kernel{
    void softmax_kernel_cpu(const Tensor& t_input, void* ptr) {
        CHECK(t_input.device_type() == base::DeviceType::CPU) << "softamx_kenerl_cpu tensor has different device type \n";

        const float* input_data = t_input.data<float>();

        arma::fvec input_fvec(const_cast<float*>(input_data), t_input.size(), false, true);
        auto max_value = *std::max_element(input_data, input_data + t_input.size());
        input_fvec = arma::exp(input_fvec - max_value);
        float sum = arma::sum(input_fvec);
        input_fvec /= sum;
    }
}