#include <glog/logging.h>
#include "rmsnorm_kernel_cpu.h"

namespace kernel {
    void rmsnorm_kernel_cpu(const Tensor& t_input, const Tensor& t_weight, Tensor& t_output, void* ptr) {
        CHECK(t_input.device_type() == t_weight.device_type() && t_input.device_type() == t_output.device_type()) << "rmsnorm_kernel_cpu tensor has different device type \n";
        CHECK(t_input.size() == t_weight.size() && t_input.size() == t_output.size()) << "rmsnorm_kernel_cpu tensor has wrong size \n";
        
        const float* input = t_input.data<float>();
        const float* weight = t_weight.data<float>();
        float* output = t_output.data<float>();

        const int32_t dim_size = static_cast<int32_t>(t_input.size());
   
        arma::fvec input_tensor(const_cast<float*>(input), dim_size, false, true);
        arma::fvec out_tensor((output), dim_size, false, true);
        arma::fvec weight_tensor(const_cast<float*>(weight), dim_size, false, true);

        const float eps = 1e-5;
 
        const float mean = arma::as_scalar(arma::mean(arma::pow(input_tensor, 2))) + eps;
        const float rsqrt = 1.f / std::sqrt(mean);
        out_tensor = weight_tensor % (rsqrt * input_tensor);
    }

}


