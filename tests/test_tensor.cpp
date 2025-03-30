#include <cstddef>
#include <gtest/gtest.h>
#include <random>
#include "base.h"
#include "tensor.h"
#include "alloc_cpu.h" 
#include "alloc_gpu.h"

TEST(test_tensor, to_cuda) {
    auto allocator_gpu = GpuAllocFactory::get_instance();
    auto allocator_cpu = CpuAllocFactory::get_instance();
    {   
        const size_t size = 1024 * 1024 * 1024;
        //Tensor t_1{size, base::DateType::DATA_FP32, allocator_gpu};
        Tensor t_1{size, base::DateType::DATA_FP32, allocator_cpu};
        float* ptr = t_1.data<float>();
        *(ptr) = static_cast<float>(0);
        *(ptr + size - 1) = static_cast<float>(size);


        const float eps = 1e-5;
        t_1.to_cuda(nullptr);
        float* cuda_ptr = t_1.data<float>();

        EXPECT_TRUE(t_1.byte_size() == 4 * size);
        EXPECT_TRUE(t_1.device_type() == base::DeviceType::GPU);
        EXPECT_TRUE(cuda_ptr != nullptr);
    }
}

TEST(test_tensor, to_cpu) {
    auto allocator_gpu = GpuAllocFactory::get_instance();
    auto allocator_cpu = CpuAllocFactory::get_instance();
    {   
        const size_t size = 1024;

        Tensor t_1{size, base::DateType::DATA_FP32, allocator_cpu};
        float* ptr = t_1.data<float>();

        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_real_distribution<float> dist(0.f, 1.f);
        float expected_data[size];

        for(size_t i = 0; i < size; ++i) {
            float v = static_cast<float>(dist(mt));
            *(ptr + i) = v;
            expected_data[i] = v;
        }

        //cuda指针不能操作
        t_1.to_cuda(nullptr);
        t_1.to_cpu();
       
        float* cpu_ptr = t_1.data<float>();   
        EXPECT_TRUE(t_1.byte_size() == 4 * size);
        EXPECT_TRUE(t_1.device_type() == base::DeviceType::CPU);
        EXPECT_TRUE(cpu_ptr != nullptr);

        const float eps = 1e-5;
        for(int i = 0; i < size; ++i) {
            EXPECT_NEAR(expected_data[i], *(cpu_ptr + i), eps);
        }
    }

}

TEST(test_tensor, external_date_test) {
    auto allocator_cpu = CpuAllocFactory::get_instance();
    {
        const size_t size = 4;
        Tensor t_1{size, base::DateType::DATA_FP32, allocator_cpu};
        float* ptr = t_1.data<float>();
        for(int i = 0; i < 4; ++i) {
            *(ptr + i) = static_cast<float>(i + 1);
        }
        float expected_data[4]{1,2,3,4}; 
        {
            Tensor t_2{size, base::DateType::DATA_FP32, allocator_cpu, ptr};
            EXPECT_TRUE(t_2.is_from_external());
            float* ptr_2 = t_2.data<float>();
            const float eps = 1e-5;
            for(int i = 0; i < 4; ++i) {
                EXPECT_NEAR(expected_data[i], *(ptr_2 + i), eps);
            }
        }
        const float eps = 1e-5;
        for(int i = 0; i < 4; ++i) {
            EXPECT_NEAR(expected_data[i], *(ptr + i), eps);
        }
    }
}