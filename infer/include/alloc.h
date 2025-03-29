#ifndef INFER_ALLOC_H
#define INFER_ALLOC_H

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include "base.h"

class DeviceAlloc{
public:
    DeviceAlloc(base::DeviceType type): device_alloc_type_(type) {};
    ~DeviceAlloc() = default;
    virtual void* allocate(size_t size) const = 0;
    virtual void deallocate(void* ptr) const = 0;
    void smemcpy(const void* src_ptr, void* dest_ptr, size_t byte_size, base::DataTransMode mode, void* stream = nullptr, bool need_sync = false);
    base::DeviceType device_type() const{
        return device_alloc_type_;
    }

    void memset_zero(void* ptr, size_t byte_size) {
        if(device_alloc_type_ == base::DeviceType::CPU) {
            memset(ptr, 0, byte_size);
        }
    }

private:
    base::DeviceType device_alloc_type_;
};

#endif


