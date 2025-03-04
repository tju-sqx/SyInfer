#include <gtest/gtest.h>
#include "tensor.h"
#include "alloc_cpu.h"

TEST(test_rmsnorm, basic_test){
    auto only_allocater = CpuAllocFactory::get_instance();
    Tensor tensor_0{24, base::DateType::DATA_FP32, only_allocater};
} 