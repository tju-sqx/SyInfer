#include <gtest/gtest.h>
#include "tensor.h"
#include "alloc_cpu.h" 

TEST(test_tensor, external_date_test) {
    auto allocator_cpu = CpuAllocFactory::get_instance();
    {
        const size_t size = 4;
        Tensor t_1{size, base::DateType::DATA_FP32, allocator_cpu};
        t_1.create();
        float* ptr = t_1.data<float>();
        for(int i = 0; i < 4; ++i) {
            *(ptr + i) = static_cast<float>(i + 1);
        }
        float expected_data[4]{1,2,3,4}; 
        {
            Tensor t_2{size, base::DateType::DATA_FP32, allocator_cpu, ptr};
            EXPECT_TRUE(t_2.is_from_external());
            EXPECT_TRUE(t_2.create());
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