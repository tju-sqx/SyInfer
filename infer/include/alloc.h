#ifndef INFER_ALLOC_H
#define INFER_ALLOC_H

#include <cstddef>
#include <cstdlib>
#include "base.h"

class DeviceAlloc{
public:
    DeviceAlloc(base::DeviceType type): device_alloc_type_(type) {};
    ~DeviceAlloc() = default;
    virtual void* allocate(size_t size) = 0;
    virtual void deallocate(void* ptr) = 0;
    
    base::DeviceType device_type() const{
        return device_alloc_type_;
    }

private:
    base::DeviceType device_alloc_type_;
};

#endif


