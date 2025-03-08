#include <gtest/gtest.h>
#include "swiglu_kernel_cpu.h"
#include "tensor.h"
#include "alloc_cpu.h"

TEST(test_swiglu, cpu_basic_test) {
    auto allocator_cpu = CpuAllocFactory::get_instance();
    {
        const size_t size = 4;

        Tensor t_input_1{size, base::DateType::DATA_FP32, allocator_cpu};
        Tensor t_input_2{size, base::DateType::DATA_FP32, allocator_cpu};
        Tensor output{size, base::DateType::DATA_FP32, allocator_cpu};
        
        t_input_1.create();
        t_input_2.create();
        output.create();

        float expected_output[4];
        float inputs_1[4]{1.0f, 0.0f, -1.0f, 2.0f};
        float inputs_2[4]{0.5f, 1.0f, 1.5f, 2.0f};

        for (int i = 0; i < size; ++i) {
            float sigmoid = 1.0f / (1.0f + std::exp(-inputs_1[i]));
            expected_output[i] = inputs_2[i] * sigmoid * inputs_1[i];
        }

        auto input_1_data = t_input_1.data<float>();
        auto input_2_data = t_input_2.data<float>();
       
        std::memcpy(input_1_data, inputs_1, size * sizeof(float));
        std::memcpy(input_2_data, inputs_2, size * sizeof(float));

        kernel::swiglu_kernel_cpu(t_input_1, t_input_2, output, nullptr);

        const float* output_data = output.data<float>();
        const float eps = 1e-5;
        for(size_t i = 0; i < size; ++i){
            EXPECT_NEAR(output_data[i], expected_output[i], eps);
        }
    }

    {
        const int size = 3;

        Tensor t_input_1{size, base::DateType::DATA_FP32, allocator_cpu};
        Tensor t_input_2{size, base::DateType::DATA_FP32, allocator_cpu};
        Tensor output{size, base::DateType::DATA_FP32, allocator_cpu};
        
        t_input_1.create();
        t_input_2.create();
        output.create();

        float inputs_1[size] = {100.0f, -100.0f, 0.0f};  // 极大值，极小值，零
        float inputs_2[size] = {1.0f, 1.0f, 1.0f};
        float expected_output[size];

        for (int i = 0; i < size; ++i) {
            float sigmoid = 1.0f / (1.0f + std::exp(-inputs_1[i]));
            expected_output[i] = inputs_2[i] * sigmoid * inputs_1[i];
        }

        auto input_1_data = t_input_1.data<float>();
        auto input_2_data = t_input_2.data<float>();
       
        std::memcpy(input_1_data, inputs_1, size * sizeof(float));
        std::memcpy(input_2_data, inputs_2, size * sizeof(float));

        kernel::swiglu_kernel_cpu(t_input_1, t_input_2, output, nullptr);

        const float* output_data = output.data<float>();
        const float eps = 1e-5;
        for(size_t i = 0; i < size; ++i){
            EXPECT_NEAR(output_data[i], expected_output[i], eps);
        }
    }
}