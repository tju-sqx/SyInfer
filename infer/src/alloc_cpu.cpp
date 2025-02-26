#include "alloc_cpu.h"

std::shared_ptr<CpuAlloc> CpuAllocFactory::cpu_allocator_ = nullptr;
void* CpuAlloc::allocate(size_t size) {
    return ::operator new(size);
}

void CpuAlloc::deallocate(void* ptr){
    ::operator delete(ptr);
}