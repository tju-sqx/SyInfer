#include "alloc_cpu.h"

void* CpuAlloc::allocate(size_t size) {
    return ::operator new(size);
}

void CpuAlloc::deallocate(void* ptr){
    ::operator delete(ptr);
}