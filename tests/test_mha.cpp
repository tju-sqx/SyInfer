#include <gtest/gtest.h>
#include "mha_kernel.h"
#include "tensor.h"
#include "alloc_cpu.h"
#include <cmath>
#include <memory>

class test_mha : public ::testing::Test {
protected:
    std::shared_ptr<DeviceAlloc> allocator_cpu;
    const size_t seq_len = 4;        // 序列长度
    const size_t head_num = 2;       // 头数量
    const size_t head_size = 3;      // 每个头的维度
    const size_t kv_dim = 3;         // key/value维度
    const size_t kv_mul = 2;         // key/value乘数
    const size_t layer_idx = 0;      // 层索引
    
    void SetUp() override {
        allocator_cpu = CpuAllocFactory::get_instance();
    }
};

TEST_F(test_mha, basic_test) {
    // 创建测试数据
    const size_t pos = 2;  // 当前位置，测试处理前3个token的情况
    
    // 创建query tensor (head_num * head_size)
    Tensor query_tensor{head_num * head_size, base::DateType::DATA_FP32, allocator_cpu};
    ASSERT_TRUE(query_tensor.create());
    
    // 创建key tensor (seq_len * kv_dim)
    Tensor key_tensor{seq_len * kv_dim, base::DateType::DATA_FP32, allocator_cpu};
    ASSERT_TRUE(key_tensor.create());
    
    // 创建value tensor (seq_len * kv_dim)
    Tensor value_tensor{seq_len * kv_dim, base::DateType::DATA_FP32, allocator_cpu};
    ASSERT_TRUE(value_tensor.create());
    
    // 创建score tensor (head_num * seq_len)
    Tensor score_tensor{head_num * seq_len, base::DateType::DATA_FP32, allocator_cpu};
    ASSERT_TRUE(score_tensor.create());
    
    // 创建输出 tensor (head_num * head_size)
    Tensor mha_out{head_num * head_size, base::DateType::DATA_FP32, allocator_cpu};
    ASSERT_TRUE(mha_out.create());
    
    // 初始化query数据
    float* query_ptr = query_tensor.data<float>();
    for (size_t i = 0; i < head_num * head_size; ++i) {
        query_ptr[i] = 0.1f * (i + 1);
    }
    
    // 初始化key数据
    float* key_ptr = key_tensor.data<float>();
    for (size_t i = 0; i < seq_len * kv_dim; ++i) {
        key_ptr[i] = 0.05f * (i + 1);
    }
    
    // 初始化value数据
    float* value_ptr = value_tensor.data<float>();
    for (size_t i = 0; i < seq_len * kv_dim; ++i) {
        value_ptr[i] = 0.2f * (i + 1);
    }
    
    // 设置设备类型
    base::DeviceType device_type = base::DeviceType::CPU;
    
    //执行MHA计算
    kernel::mha_kernel(pos, seq_len, kv_dim, head_num, head_size, kv_mul, layer_idx,
                       mha_out, query_tensor, key_tensor, value_tensor, score_tensor, device_type);
    
    // 验证结果
    // 这里只做基本的非零检查，因为具体数值需要手动计算或使用参考实现进行比较
    float* output_ptr = mha_out.data<float>();
    bool has_nonzero = false;
    
    for (size_t i = 0; i < head_num * head_size; ++i) {
        if (std::abs(output_ptr[i]) > 1e-6) {
            has_nonzero = true;
            break;
        }
    }
    
    EXPECT_TRUE(has_nonzero) << "Output should contain non-zero values";
}

TEST_F(test_mha, CorrectCalculation) {
    // 这个测试使用简化的输入数据，并手动计算期望的输出结果
    const size_t pos = 1;  // 只处理前两个token
    
    // 创建简化的tensors
    Tensor query_tensor{head_num * head_size, base::DateType::DATA_FP32, allocator_cpu};
    Tensor key_tensor{seq_len * kv_dim, base::DateType::DATA_FP32, allocator_cpu};
    Tensor value_tensor{seq_len * kv_dim, base::DateType::DATA_FP32, allocator_cpu};
    Tensor score_tensor{head_num * seq_len, base::DateType::DATA_FP32, allocator_cpu};
    Tensor mha_out{head_num * head_size, base::DateType::DATA_FP32, allocator_cpu};
    
    query_tensor.create();
    key_tensor.create();
    value_tensor.create();
    score_tensor.create();
    mha_out.create();
    
    // 用简单数据初始化
    float* query_ptr = query_tensor.data<float>();
    float* key_ptr = key_tensor.data<float>();
    float* value_ptr = value_tensor.data<float>();
    
    // 简化数据，便于手动验证
    // 第一个头的query [1, 1, 1]
    // 第二个头的query [2, 2, 2]
    for (size_t h = 0; h < head_num; ++h) {
        for (size_t i = 0; i < head_size; ++i) {
            query_ptr[h * head_size + i] = h + 1;
        }
    }
    
    // 所有key都设为[1, 1, 1]
    for (size_t i = 0; i < seq_len * kv_dim; ++i) {
        key_ptr[i] = 1.0f;
    }
    
    // 所有value设为位置索引
    for (size_t p = 0; p < seq_len; ++p) {
        for (size_t i = 0; i < kv_dim; ++i) {
            value_ptr[p * kv_dim + i] = p + 1;
        }
    }
    
    base::DeviceType device_type = base::DeviceType::CPU;
    
    // 执行MHA计算
    kernel::mha_kernel(pos, seq_len, kv_dim, head_num, head_size, kv_mul, layer_idx,
                       mha_out, query_tensor, key_tensor, value_tensor, score_tensor, device_type);
    
    // 手动计算期望结果
    // 对于这个简化模型：
    // 1. query和key的点积都是head_size (因为全是1)
    // 2. 缩放因子是1/sqrt(head_size)
    // 3. 所有得分会通过softmax，因为都一样，所以每个得分应该是1/(pos+1)
    // 4. 最后，每个头的输出应该是权重加权的value之和
    
    float scale = 1.0f / std::sqrt(static_cast<float>(head_size));
    float score_value = scale * head_size; // 未归一化的得分值
    float softmax_value = 0.5f; // 两个相同值的softmax结果是各0.5
    
    // 验证输出
    float* output_ptr = mha_out.data<float>();
    
    // 第一个头的输出应该是(0.5*1 + 0.5*2) = 1.5
    // 第二个头的输出应该是(0.5*1 + 0.5*2) = 1.5 (query不同，但对这个简化模型没影响)
    const float expected_value = 1.5f;
    const float eps = 1e-5f;
    
    for (size_t h = 0; h < head_num; ++h) {
        for (size_t i = 0; i < head_size; ++i) {
            EXPECT_NEAR(output_ptr[h * head_size + i], expected_value, eps)
                << "Output mismatch at head " << h << ", position " << i;
        }
    }
}

TEST_F(test_mha, ZeroPosition) {
    // 测试pos=0的边界情况，即只处理一个token
    const size_t pos = 0;
    
    Tensor query_tensor{head_num * head_size, base::DateType::DATA_FP32, allocator_cpu};
    Tensor key_tensor{seq_len * kv_dim, base::DateType::DATA_FP32, allocator_cpu};
    Tensor value_tensor{seq_len * kv_dim, base::DateType::DATA_FP32, allocator_cpu};
    Tensor score_tensor{head_num * seq_len, base::DateType::DATA_FP32, allocator_cpu};
    Tensor mha_out{head_num * head_size, base::DateType::DATA_FP32, allocator_cpu};
    
    query_tensor.create();
    key_tensor.create();
    value_tensor.create();
    score_tensor.create();
    mha_out.create();
    
    // 用1.0初始化所有数据
    float* query_ptr = query_tensor.data<float>();
    float* key_ptr = key_tensor.data<float>();
    float* value_ptr = value_tensor.data<float>();
    
    for (size_t i = 0; i < head_num * head_size; ++i) {
        query_ptr[i] = 1.0f;
    }
    
    for (size_t i = 0; i < seq_len * kv_dim; ++i) {
        key_ptr[i] = 1.0f;
    }
    
    for (size_t i = 0; i < seq_len * kv_dim; ++i) {
        value_ptr[i] = 1.0f;
    }
    
    base::DeviceType device_type = base::DeviceType::CPU;
    
    // 执行MHA计算
    kernel::mha_kernel(pos, seq_len, kv_dim, head_num, head_size, kv_mul, layer_idx,
                       mha_out, query_tensor, key_tensor, value_tensor, score_tensor, device_type);
    
    // 因为只处理一个token，所以得分softmax就是1.0
    // 输出应该等于value的值
    float* output_ptr = mha_out.data<float>();
    const float expected_value = 1.0f;
    const float eps = 1e-5f;
    
    for (size_t i = 0; i < head_num * head_size; ++i) {
        EXPECT_NEAR(output_ptr[i], expected_value, eps);
    }
}