#include <gtest/gtest.h>
#include "emb_kernel_cpu.h"
#include "tensor.h"
#include "alloc_cpu.h"

TEST(test_emb, basic_test_cpu) {
    auto allocator_cpu = CpuAllocFactory::get_instance();
    {
        const size_t vocab_size = 3;
        const size_t embedding_dim = 4;
        const size_t input_size = 2;

        // 创建Tensor对象
        Tensor t_input{input_size, base::DateType::DATA_FP32, allocator_cpu};
        Tensor t_weight{vocab_size, embedding_dim, base::DateType::DATA_FP32, allocator_cpu};
        Tensor t_output{input_size, embedding_dim, base::DateType::DATA_FP32, allocator_cpu};
        
        // 填充输入数据
        float input_data[input_size] = {0.0f, 2.0f};  // token ids
        float weight_data[vocab_size * embedding_dim] = {
            1.0f, 2.0f, 3.0f, 4.0f,  // token 0
            5.0f, 6.0f, 7.0f, 8.0f,  // token 1
            9.0f, 10.0f, 11.0f, 12.0f // token 2
        };
        float expected_output[input_size * embedding_dim] = {
            1.0f, 2.0f, 3.0f, 4.0f,  // token 0
            9.0f, 10.0f, 11.0f, 12.0f // token 2
        };


        std::memcpy(t_input.data<float>(), input_data, input_size * sizeof(float));
        std::memcpy(t_weight.data<float>(), weight_data, vocab_size * embedding_dim * sizeof(float));

        // 调用算子
        kernel::emb_kernel_cpu(t_input, t_weight, t_output, nullptr);

        // 验证结果
        const float* output_data = t_output.data<float>();
        const float eps = 1e-5;
        
        for(size_t i = 0; i < input_size * embedding_dim; ++i){
            EXPECT_NEAR(output_data[i], expected_output[i], eps);
        }
    }
}

TEST(test_emb, DISABLED_cpu_exception_test){
    auto allocator_cpu = CpuAllocFactory::get_instance();
    {
        // 准备测试数据
        const size_t vocab_size = 3;
        const size_t embedding_dim = 4;
        const size_t input_size = 2;

        // 创建Tensor对象
        Tensor t_input{input_size, base::DateType::DATA_FP32, allocator_cpu};
        Tensor t_weight{vocab_size, embedding_dim, base::DateType::DATA_FP32, allocator_cpu};
        Tensor t_output{input_size, embedding_dim, base::DateType::DATA_FP32, allocator_cpu};
        Tensor t_empty{0, base::DateType::DATA_FP32, allocator_cpu};

        // 测试空输入
        EXPECT_EXIT(kernel::emb_kernel_cpu(t_empty, t_weight, t_output, nullptr), 
        ::testing::KilledBySignal(SIGABRT), ".*");
        EXPECT_EXIT(kernel::emb_kernel_cpu(t_input, t_empty, t_output, nullptr), 
        ::testing::KilledBySignal(SIGABRT), ".*");
    }
}