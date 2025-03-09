#include <gtest/gtest.h>
#include "softmax_kernel_cpu.h"
#include "tensor.h"
#include "alloc_cpu.h"

TEST(test_softmax, basic_test_cpu) {
    auto allocator_cpu = CpuAllocFactory::get_instance();
        {
        const size_t size = 4;

        Tensor t_input{size, base::DateType::DATA_FP32, allocator_cpu};
        Tensor output{size, base::DateType::DATA_FP32, allocator_cpu};
        
        t_input.create();
        output.create();

        float inputs[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        float expected_output[4];
        
        // 计算期望值
        float max_val = *std::max_element(inputs, inputs + size);
        float sum = 0.0f;
        for (int i = 0; i < size; ++i) {
            expected_output[i] = std::exp(inputs[i] - max_val);
            sum += expected_output[i];
        }
        for (int i = 0; i < size; ++i) {
            expected_output[i] /= sum;
        }

        // 填充输入数据
        auto input_data = t_input.data<float>();
        std::memcpy(input_data, inputs, size * sizeof(float));

        // 调用算子
        kernel::softmax_kernel_cpu(t_input, nullptr);

        // 验证结果
        const float* output_data = t_input.data<float>();
        const float eps = 1e-5;
        float sum_output = 0.0f;
        
        for(size_t i = 0; i < size; ++i){
            EXPECT_NEAR(output_data[i], expected_output[i], eps);
            sum_output += output_data[i];
            EXPECT_GE(output_data[i], 0.0f);
            EXPECT_LE(output_data[i], 1.0f);
        }
        EXPECT_NEAR(sum_output, 1.0f, eps);
    }

    //all zero
        {
        const size_t size = 3;

        Tensor t_input{size, base::DateType::DATA_FP32, allocator_cpu};
        Tensor output{size, base::DateType::DATA_FP32, allocator_cpu};
        
        t_input.create();
        output.create();

        float inputs[3] = {0.0f, 0.0f, 0.0f};
        const float expected_val = 1.0f / 3.0f;

        auto input_data = t_input.data<float>();
        std::memcpy(input_data, inputs, size * sizeof(float));

        kernel::softmax_kernel_cpu(t_input, nullptr);

        const float* output_data = t_input.data<float>();
        const float eps = 1e-5;
        
        for(size_t i = 0; i < size; ++i){
            EXPECT_NEAR(output_data[i], expected_val, eps);
        }
    }

    //large num
    {
        const size_t size = 2;

        Tensor t_input{size, base::DateType::DATA_FP32, allocator_cpu};
        Tensor output{size, base::DateType::DATA_FP32, allocator_cpu};
        
        t_input.create();
        output.create();

        float inputs[2] = {1000.0f, 1001.0f};
        const float diff = 1001.0f - 1000.0f;
        const float expected = 1.0f / (1.0f + std::exp(-diff));

        auto input_data = t_input.data<float>();
        std::memcpy(input_data, inputs, size * sizeof(float));

        kernel::softmax_kernel_cpu(t_input, nullptr);

        const float* output_data = t_input.data<float>();
        const float eps = 1e-5;
        
        EXPECT_NEAR(output_data[0], 1.0f - expected, eps);
        EXPECT_NEAR(output_data[1], expected, eps);
    }
}