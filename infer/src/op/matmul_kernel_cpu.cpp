#include "matmul_kernel_cpu.h"
#include <glog/logging.h>

namespace kernel {
    void matmul_kernel_cpu(const Tensor& input_tensor, const Tensor& weight_tensor, const Tensor& output_tensor, float scale) {
        CHECK(input_tensor.device_type() == base::DeviceType::CPU) << "input_tensor is wrong device type in matmul_kernel_cpu func \n";     
        CHECK(output_tensor.device_type() == base::DeviceType::CPU) << "output_tensor is wrong device type in matmul_kernel_cpu func \n";
        CHECK(weight_tensor.device_type() == base::DeviceType::CPU) << "weight_tensor is wrong device type in matmul_kernel_cpu func \n";
        CHECK(input_tensor.empty() == false) << "input_tensor is empty in matmul__kernel_cpu func \n";
        CHECK(output_tensor.empty() == false) << "output_tensor is empty in matmul__kernel_cpu func \n";
        CHECK(weight_tensor.empty() == false) << "weight_tensor is empty in matmul__kernel_cpu func \n";

        size_t i_dim_0 = 1;
        size_t i_dim_1 = 1;
        if(input_tensor.dim() == 2) {
            i_dim_0 = input_tensor.get_dim(0);
            i_dim_1 = input_tensor.get_dim(1);
        } else if(input_tensor.dim() == 1){
            i_dim_0 = input_tensor.get_dim(0);
        } else {
            LOG(FATAL) << "input_tensor has wrong dims in matmul func\n";
        }
        size_t w_dim_0 = weight_tensor.get_dim(0);
        size_t w_dim_1 = weight_tensor.dim() > 1? weight_tensor.get_dim(1) : 1;
        if(weight_tensor.dim() > 2) {
            LOG(FATAL) << "weight_tensor has wrong dims in matmul func\n";
        }
        CHECK_EQ(w_dim_1, i_dim_0) << "input_tensor dim dismatch weight_tensor dim in matmul func\n";
        CHECK_EQ(output_tensor.size(), w_dim_0 * i_dim_1) <<"output_tensor dim does not equal to result in matmul func \n";
        
        const float* input_ptr = input_tensor.data<float>();
        const float* output_ptr = output_tensor.data<float>();
        const float* weight_ptr = weight_tensor.data<float>();

        arma::fmat input_mat(const_cast<float*>(input_ptr), i_dim_1, i_dim_0, false, true);
        arma::fmat output_mat(const_cast<float*>(output_ptr), i_dim_1, w_dim_0, false, true);
        arma::fmat weight_mat(const_cast<float*>(weight_ptr), w_dim_1, w_dim_0, false, true);

        output_mat = (input_mat * weight_mat) * scale;
    }
}