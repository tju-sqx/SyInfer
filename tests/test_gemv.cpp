#include <gtest/gtest.h>
#include "matmul_kernel_cpu.h"
#include "tensor.h"
#include "alloc_cpu.h"

TEST(test_gemv, basic_test_cpu) {
    auto allocator_cpu = CpuAllocFactory::get_instance();
    {
        size_t dim = 3;
        Tensor input{dim, base::DateType::DATA_FP32, allocator_cpu};
        Tensor weight{dim, dim, base::DateType::DATA_FP32, allocator_cpu};
        Tensor output{dim, base::DateType::DATA_FP32, allocator_cpu};

        input.create();
        output.create();
        weight.create();

        for(size_t i = 0; i < dim; ++i) {
            *(input.data<float>() + i) = static_cast<float>(i + 1);
        }

        for(size_t i = 0; i < dim * dim; ++i) {
            *(weight.data<float>() + i) = static_cast<float>(i + 1);
        }

        float expected_output[3] = {14.0f, 32.0f, 50.0f};

        kernel::matmul_kernel_cpu(input, weight, output);
        const float eps = 1e-5;
        for(size_t i = 0; i < 3; ++i) {
            EXPECT_NEAR(*(output.data<float>() + i), expected_output[i], eps) << "i: "<<i<<"\n";
        }                                                                                                                                             
    }   
}