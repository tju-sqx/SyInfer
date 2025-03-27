#include <gtest/gtest.h>
#include "base.h"
#include "scale_sum_kernel.h"
#include "tensor.h"
#include "alloc_cpu.h"

TEST(test_scale_sum, basic_test_cpu) {
    auto allocator_cpu = CpuAllocFactory::get_instance();
    {
        // 准备测试数据
        const size_t head_size = 4;
        const size_t pos = 2;
        const size_t step = head_size;
        
        // 创建输入张量
        Tensor score_tensor{pos + 1, base::DateType::DATA_FP32, allocator_cpu};
        Tensor value_tensor {(step) * (pos + 1), base::DateType::DATA_FP32, allocator_cpu};
        Tensor output_tensor{head_size, base::DateType::DATA_FP32, allocator_cpu};
        
        score_tensor.create();
        value_tensor.create();
        output_tensor.create();
        
        // 填充输入数据
        float score_data[pos + 1] = {0.5f, 0.3f, 0.2f};
        float value_data[(pos + 1) * step] = {
            1.0f, 2.0f, 3.0f, 4.0f,  // p=0
            5.0f, 6.0f, 7.0f, 8.0f,  // p=1
            9.0f, 10.0f, 11.0f, 12.0f // p=2
        };
        float output_data[head_size] = {0.0f, 0.0f, 0.0f, 0.0f};  // 初始化为0
        
        std::memcpy(score_tensor.data<float>(), score_data, (pos + 1) * sizeof(float));
        std::memcpy(value_tensor.data<float>(), value_data, (pos + 1) * step * sizeof(float));
        std::memcpy(output_tensor.data<float>(), output_data, head_size * sizeof(float));
        
        // 设置设备类型
        base::DeviceType device_type = base::DeviceType::CPU;
        score_tensor.set_device_type(device_type);
        value_tensor.set_device_type(device_type);
        output_tensor.set_device_type(device_type);
        
        Tensor value_vec = Tensor{head_size, base::DateType::DATA_FP32, nullptr, (const_cast<float*>(value_tensor.data<float>()))};
        value_vec.create();
        // 调用被测函数
        kernel::scale_sum_kernel(score_tensor, value_vec, output_tensor, head_size, pos, step);
        
        // 手动计算期望结果
        float expected_output[head_size] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (size_t p = 0; p <= pos; ++p) {
            for (size_t h = 0; h < head_size; ++h) {
                expected_output[h] += score_data[p] * value_data[p * step + h];
            }
        }
        
        // 验证结果
        const float* actual_output = output_tensor.data<float>();
        const float eps = 1e-5;
        
        for (size_t i = 0; i < head_size; ++i) {
            EXPECT_NEAR(expected_output[i], actual_output[i], eps);
        }
    }
}