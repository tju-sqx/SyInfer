#ifndef INFER_ALLOC_H
#define INFER_ALLOC_H

#include <cstddef>
#include <cstdlib>
#include "base.h"

class DeviceAlloc{
public:
    DeviceAlloc(base::DeviceAllocType type): device_alloc_type_(type) {};
    ~DeviceAlloc() = default;
    virtual void* allocate(size_t size) = 0;
    virtual void deallocate(void* ptr) = 0;
private:
    base::DeviceAllocType device_alloc_type_;
};

#endif


