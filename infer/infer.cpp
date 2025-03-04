#include <iostream>
#include "alloc.h"
#include "alloc_cpu.h"
#include "buffer.h"
#include "tensor.h"
#include "rmsnorm_kernel_cpu.h"

#define LOG(M) std::cout<<M<<std::endl;
int main(){

    CpuAlloc allocator{};
    auto ptr1 = allocator.allocate(32);
    auto ptr2 = allocator.allocate(64);

    std::cout<<"ptr1 addr: "<<ptr1<<std::endl;
    std::cout<<"ptr2 addr: "<<ptr2<<std::endl;
    allocator.deallocate(ptr1);
    allocator.deallocate(ptr2);
    std::cout<<"is delete or not"<<(!ptr1)<<std::endl;
    std::cout<<"is delete or not"<<(!ptr2)<<std::endl;

    std::cout<<"use instance"<<std::endl;


    auto only_allocater = CpuAllocFactory::get_instance();
    auto ptr3 = only_allocater->allocate(32);
    auto ptr4 = only_allocater->allocate(64);
    std::cout<<"ptr3 addr: "<<ptr3<<std::endl;
    std::cout<<"ptr4 addr: "<<ptr4<<std::endl;
    only_allocater->deallocate(ptr3);
    only_allocater->deallocate(ptr4);
    std::cout<<"is delete or not"<<(!ptr3)<<std::endl;
    std::cout<<"is delete or not"<<(!ptr4)<<std::endl;

    auto buffer_ptr = std::make_shared<Buffer>(32, only_allocater, base::DeviceType::CPU);
    std::cout<<"Buffer size "<<buffer_ptr->size()<<std::endl;
    Buffer buffer{32, only_allocater, base::DeviceType::CPU};

    {
        Tensor tensor_0{24, base::DateType::DATA_FP32, only_allocater};
        //tensor = tensor;
        Tensor tensor_1{24, base::DateType::DATA_FP32, only_allocater};
        Tensor tensor_2{24, base::DateType::DATA_FP32, only_allocater};
        tensor_0.create();
        
        auto ptr1 = tensor_0.data<float>();

        for(int i = 0; i < 24; ++i) {
            *(ptr1) = static_cast<float>(1);
            ptr1++;
            
        }
        std::cout<<"complete"<<std::endl;
        (void) kernel::rmsnorm_kernel_cpu(tensor_0, tensor_0, tensor_0, nullptr);
    }
    return 0;
}