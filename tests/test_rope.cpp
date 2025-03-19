#include <gtest/gtest.h>
#include "rope_kernel_cpu.h"
#include "tensor.h"
#include "alloc_cpu.h"
#include <cmath>
TEST(test_rope, sin_cos_calc_test) {
    {
        const size_t head_size = 4;
        const size_t max_len_size = 3;
        
        // 分配缓存
        float sin_cache[head_size * max_len_size];
        float cos_cache[head_size * max_len_size];

        // 调用函数
        kernel::sin_cos_calc(head_size, max_len_size, sin_cache, cos_cache);

        // 验证结果
        for (size_t pos = 0; pos < max_len_size; ++pos) {
            for (size_t head_dim = 0; head_dim < head_size; ++head_dim) {
                // 计算期望值
                float frec = 1.0f / std::pow(10000.0f, static_cast<float>(head_dim) / static_cast<float>(head_size));
                float rotate = static_cast<float>(pos) * frec;
                float expected_sin = std::sin(rotate);
                float expected_cos = std::cos(rotate);

                // 获取实际值
                float actual_sin = sin_cache[pos * head_size + head_dim];
                float actual_cos = cos_cache[pos * head_size + head_dim];

                // 验证
                EXPECT_NEAR(actual_sin, expected_sin, 1e-5)
                    << "Position: " << pos << ", Head dim: " << head_dim;
                EXPECT_NEAR(actual_cos, expected_cos, 1e-5)
                    << "Position: " << pos << ", Head dim: " << head_dim;
            }
        }
    }

    // edge
    {
        const size_t head_size = 1;
        const size_t max_len_size = 1;
        
        float sin_cache[head_size * max_len_size];
        float cos_cache[head_size * max_len_size];

        kernel::sin_cos_calc(head_size, max_len_size, sin_cache, cos_cache);

        // 验证最小维度情况
        EXPECT_NEAR(sin_cache[0], 0.0f, 1e-5);
        EXPECT_NEAR(cos_cache[0], 1.0f, 1e-5);
    }

    {
        const size_t head_size = 128;
        const size_t max_len_size = 1024;
        
        std::vector<float> sin_cache(head_size * max_len_size);
        std::vector<float> cos_cache(head_size * max_len_size);

        kernel::sin_cos_calc(head_size, max_len_size, sin_cache.data(), cos_cache.data());

        // 验证大输入情况
        for (size_t pos = 0; pos < max_len_size; pos += 100) {
            for (size_t head_dim = 0; head_dim < head_size; head_dim += 16) {
                float frec = 1.0f / std::pow(10000.0f, static_cast<float>(head_dim) / static_cast<float>(head_size));
                float rotate = static_cast<float>(pos) * frec;
                float expected_sin = std::sin(rotate);
                float expected_cos = std::cos(rotate);

                EXPECT_NEAR(sin_cache[pos * head_size + head_dim], expected_sin, 1e-5);
                EXPECT_NEAR(cos_cache[pos * head_size + head_dim], expected_cos, 1e-5);
            }
        }
    }
}

TEST(test_rope, basic_test_cpu) {
    auto allocator_cpu = CpuAllocFactory::get_instance();
    {
        // 准备测试数据
        const size_t dim_size = 4;
        const size_t k_size = 2;
        const size_t head_size = 2;
        const size_t pos = 1;

        // 创建Tensor对象
        Tensor pos_input{1, base::DateType::DATA_INT8, allocator_cpu};
        Tensor k_input{dim_size, base::DateType::DATA_FP32, allocator_cpu};
        Tensor q_input{dim_size, base::DateType::DATA_FP32, allocator_cpu};
        Tensor sin_cache{head_size * (pos + 1), base::DateType::DATA_FP32, allocator_cpu};
        Tensor cos_cache{head_size * (pos + 1), base::DateType::DATA_FP32, allocator_cpu};
        
        pos_input.create();
        k_input.create();
        q_input.create();
        sin_cache.create();
        cos_cache.create();   
        // 填充输入数据
        size_t pos_data = pos;
        float k_data[dim_size] = {1.0f, 2.0f, 3.0f, 4.0f};
        float q_data[dim_size] = {5.0f, 6.0f, 7.0f, 8.0f};
        float sin_data[head_size * (pos + 1)] = {0.0f, 1.0f, 0.5f, 0.866f};  // sin(0), sin(60), sin(30), sin(60)
        float cos_data[head_size * (pos + 1)] = {1.0f, 0.0f, 0.866f, 0.5f};  // cos(0), cos(60), cos(30), cos(60)

        std::memcpy(pos_input.data<size_t>(), &pos_data, sizeof(size_t));
        std::memcpy(k_input.data<float>(), k_data, dim_size * sizeof(float));
        std::memcpy(q_input.data<float>(), q_data, dim_size * sizeof(float));
        std::memcpy(sin_cache.data<float>(), sin_data, head_size * (pos + 1) * sizeof(float));
        std::memcpy(cos_cache.data<float>(), cos_data, head_size * (pos + 1) * sizeof(float));

        // 调用算子
        kernel::rope_kernel_cpu(dim_size, k_size, head_size, pos_input, k_input, q_input, sin_cache, cos_cache);

        // 验证结果
        const float* k_output = k_input.data<float>();
        const float* q_output = q_input.data<float>();
        const float eps = 1e-5;

        // 预期结果计算
        float expected_k[dim_size] = {
            1.0f * 0.866f - 2.0f * 0.5f,  // k[0] * cos(30) - k[1] * sin(30)
            1.0f * 0.5f + 2.0f * 0.866f,  // k[0] * sin(30) + k[1] * cos(30)
            3.0f,
            4.0f
        };

        float expected_q[dim_size] = {
            5.0f * 0.866f - 6.0f * 0.5f,  // q[0] * cos(30) - q[1] * sin(30)
            5.0f * 0.5f + 6.0f *  0.866f,  // q[0] * sin(30) + q[1] * cos(30)
            7.0f * 0.866f - 8.0f * 0.5f,  // q[2] * cos(30) - q[3] * sin(30)
            7.0f * 0.5f + 8.0f * 0.866f   // q[2] * sin(30) + q[3] * cos(30)
        };


        for(size_t i = 0; i < dim_size; ++i){
            EXPECT_NEAR(k_output[i], expected_k[i], eps);
        }

        // 验证q_input
        for(size_t i = 0; i < dim_size; ++i){
            EXPECT_NEAR(q_output[i], expected_q[i], eps) << "i: "<<i<<"\n";
        }
    }
}