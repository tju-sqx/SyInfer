#include "alloc_cpu.h"

//std::shared_ptr<CpuAlloc> CpuAllocFactory::cpu_allocator_ = nullptr;
void* CpuAlloc::allocate(size_t size) const {
    return ::operator new(size);
}

void CpuAlloc::deallocate(void* ptr) const {
    ::operator delete(ptr);
}