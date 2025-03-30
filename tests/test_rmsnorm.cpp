#include <gtest/gtest.h>
#include "rmsnorm_kernel_cpu.h"
#include "tensor.h"
#include "alloc_cpu.h"

void assign(float* tensor_data, float* assign_value, const size_t size) {
    for(auto i = 0; i < size; ++i){
        *(tensor_data + i ) = *(assign_value +i);
    }
}

TEST(test_rmsnorm, cpu_basic_test){
    auto allocator_cpu = CpuAllocFactory::get_instance();
    {
        const size_t size = 4;

        Tensor input{size, base::DateType::DATA_FP32, allocator_cpu};
        Tensor output{size, base::DateType::DATA_FP32, allocator_cpu};
        Tensor weight{size, base::DateType::DATA_FP32, allocator_cpu};

        float inputs[4]{1.0f, 2.0f, 3.0f, 4.0f};
        float weights[4]{0.5f, 0.5f,0.5f, 0.5f};

        auto input_data = input.data<float>();
        auto weight_data = weight.data<float>();
        assign(input_data, inputs, size);
        assign(weight_data, weights, size);
        
        kernel::rmsnorm_kernel_cpu(input, weight, output, nullptr);

        const float* output_data = output.data<float>();
        const float eps = 1e-5;
        const float expected_mean = (1.0f + 4.0f + 9.0f + 16.0f) / 4.0f + eps;
        const float expected_rsqrt = 1.0f / std::sqrt(expected_mean);
        EXPECT_NEAR(output_data[0], 0.5f * expected_rsqrt * inputs[0], eps);
        EXPECT_NEAR(output_data[1], 0.5f * expected_rsqrt * inputs[1], eps);
        EXPECT_NEAR(output_data[2], 0.5f * expected_rsqrt * inputs[2], eps);
        EXPECT_NEAR(output_data[3], 0.5f * expected_rsqrt * inputs[3], eps);
    }

    //zero input
    {
        const size_t size = 4;

        Tensor input{size, base::DateType::DATA_FP32, allocator_cpu};
        Tensor output{size, base::DateType::DATA_FP32, allocator_cpu};
        Tensor weight{size, base::DateType::DATA_FP32, allocator_cpu};

        auto input_data = input.data<float>();
        auto weight_data = weight.data<float>();
        std::fill(input_data, input_data + size, 0.0f);
        std::fill(weight_data, weight_data + size, 1.0f);

        kernel::rmsnorm_kernel_cpu(input, weight, output, nullptr);
        const float eps = 1e-5;
        const float* output_data = output.data<float>();
        for(int i = 0; i < size; ++i) {
            EXPECT_NEAR(output_data[i], 0.0f, eps);
        }
    }
    
    //large input
    {
        const size_t size = 10000;

        Tensor input{size, base::DateType::DATA_FP32, allocator_cpu};
        Tensor output{size, base::DateType::DATA_FP32, allocator_cpu};
        Tensor weight{size, base::DateType::DATA_FP32, allocator_cpu};

        auto input_data = input.data<float>();
        auto weight_data = weight.data<float>();
        
        for(int i = 0; i < size; ++i) {
            input_data[i] = static_cast<float>(i);
        }
        std::fill(weight_data, weight_data + size, 0.5f);

        kernel::rmsnorm_kernel_cpu(input, weight, output, nullptr);
        const float eps = 1e-5;
        float sum_squares = 0.0f;
        for(int i = 0; i < size; ++i){
            sum_squares += input_data[i] * input_data[i];
        }
        const float expect_mean = sum_squares / size + eps;
        const float expect_rsqrt = 1.f / std::sqrt(expect_mean);

        const float* output_data = output.data<float>();
        for(int i = 0; i < size; ++i) {
            EXPECT_NEAR(output_data[i], weight_data[i] * input_data[i] * expect_rsqrt , eps);
        }
    }
    
}

TEST(test_rmsnorm, DISABLED_cpu_exception_test) {
    auto allocator_cpu = CpuAllocFactory::get_instance();
    {   
        const int size_1 = 1;
        const int size_2 = 2;
        Tensor input{size_1, base::DateType::DATA_FP32, allocator_cpu};
        Tensor output{size_1, base::DateType::DATA_FP32, allocator_cpu};
        Tensor weight{size_2, base::DateType::DATA_FP32, allocator_cpu};

        EXPECT_EXIT(kernel::rmsnorm_kernel_cpu(input, weight, output, nullptr), 
                   ::testing::KilledBySignal(SIGABRT), ".*");
    }
}