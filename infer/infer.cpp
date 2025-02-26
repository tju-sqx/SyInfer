#include <iostream>
#include "alloc.h"
#include "alloc_cpu.h"
#include "buffer.h"

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

    auto buffer_ptr = std::make_shared<Buffer>(32, only_allocater, base::DeviceAllocType::CPU);
    std::cout<<"Buffer size "<<buffer_ptr->size()<<std::endl;
    Buffer buffer{32, only_allocater, base::DeviceAllocType::CPU};
    return 0;
}