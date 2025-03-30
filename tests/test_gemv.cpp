#include <gtest/gtest.h>
#include "matmul_kernel_cpu.h"
#include "tensor.h"
#include "alloc_cpu.h"
#include "alloc_gpu.h"
#include "gemv_kernel_cuda.cuh"
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
TEST(test_gemv, basic_test_gpu) {
    auto allocator_cpu = CpuAllocFactory::get_instance();
    auto allocator_gpu = GpuAllocFactory::get_instance();
    {      
        const float eps = 1e-5;
        const size_t dim = 3;
        Tensor input_cpu{dim, base::DateType::DATA_FP32, allocator_cpu};
        Tensor weight_cpu{dim, dim, base::DateType::DATA_FP32, allocator_cpu};
        Tensor output_cpu{dim, base::DateType::DATA_FP32, allocator_cpu};

        Tensor input_gpu{dim, base::DateType::DATA_FP32, allocator_cpu};
        Tensor weight_gpu{dim, dim, base::DateType::DATA_FP32, allocator_cpu};
        Tensor output_gpu{dim, base::DateType::DATA_FP32, allocator_cpu};


        for(size_t i = 0; i < dim; ++i) {
            *(input_cpu.data<float>() + i) = static_cast<float>(i + 1);
            *(input_gpu.data<float>() + i) = static_cast<float>(i + 1);
        }

        for(size_t i = 0; i < dim * dim; ++i) {
            *(weight_cpu.data<float>() + i) = static_cast<float>(i + 1);
            *(weight_gpu.data<float>() + i) = static_cast<float>(i + 1);
        }

        input_gpu.to_cuda(nullptr);
        weight_gpu.to_cuda(nullptr);
        output_gpu.to_cuda(nullptr);

        float expected_output[3] = {14.0f, 32.0f, 50.0f};

        kernel::matmul_kernel_cpu(input_cpu, weight_cpu, output_cpu);
        kernel::gemv_kernel_cuda(input_gpu, weight_gpu, output_gpu);

        input_gpu.to_cpu();
        weight_gpu.to_cpu();
        output_gpu.to_cpu();
        
        float* gpu_ptr = input_gpu.data<float>();
        for(size_t i = 0; i < dim; ++i) {
            EXPECT_NEAR(*(gpu_ptr + i),*(input_cpu.data<float>() + i), eps) << "i: "<<i<<"\n";
        }

        for(size_t i = 0; i < 3; ++i) {
            EXPECT_NEAR(*(output_gpu.data<float>() + i),*(output_cpu.data<float>() + i), eps) << "i: "<<i<<"\n";
        }                                                                                                                                             
    }  
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
TEST(test_gemv, DISABLED_cpu_exception_test) {
    auto allocator_cpu = CpuAllocFactory::get_instance();
    {
        size_t dim = 3;
        Tensor input{dim, base::DateType::DATA_FP32, allocator_cpu};
        Tensor weight{dim, 4, base::DateType::DATA_FP32, allocator_cpu};
        Tensor output{dim, base::DateType::DATA_FP32, allocator_cpu};

        for(size_t i = 0; i < dim; ++i) {
            *(input.data<float>() + i) = static_cast<float>(i + 1);
        }

        for(size_t i = 0; i < dim * dim; ++i) {
            *(weight.data<float>() + i) = static_cast<float>(i + 1);
        }

        EXPECT_EXIT(kernel::matmul_kernel_cpu(input, weight, output);, 
                   ::testing::KilledBySignal(SIGABRT), ".*");                                                                                                                                       
    }   
}