#include <glog/logging.h>
#include "swiglu_kernel_cpu.h"

void kernel::swiglu_kernel_cpu(const Tensor& t_input_1, const Tensor& t_input_2, Tensor& t_output, void* ptr){
    CHECK(t_input_1.device_type() == t_input_2.device_type() 
    && t_input_1.device_type() == t_output.device_type()) << "swiglu_kernel_cpu tensor has different device type \n";
    CHECK(t_input_1.size() == t_input_2.size() 
    && t_input_1.size() == t_output.size()) << "swiglu_kernel_cpu tensor has wrong size \n";
    
    const float* input_data_1 = t_input_1.data<float>();
    const float* input_data_2 = t_input_2.data<float>();
    float* output_data = t_output.data<float>();

    arma::fvec input_fvec_1(const_cast<float*>(input_data_1), t_input_1.size(), false, true);
    arma::fvec output_fvec(output_data, t_output.size(), false, true);
    arma::fvec input_fvec_2(const_cast<float*>(input_data_2), t_input_2.size(), false, true);

    input_fvec_1 %= (1.0f / (1.0f + arma::exp(-input_fvec_1)));
    output_fvec = input_fvec_1 % input_fvec_2;
}