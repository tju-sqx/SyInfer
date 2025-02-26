#ifndef BUFFER_H
#define BUFFER_H


#include "alloc.h"
#include <memory>

class Buffer: public base::NoCopyable, std::enable_shared_from_this<Buffer>{
public:
    explicit Buffer() = default;
    explicit Buffer(size_t size, std::shared_ptr<DeviceAlloc> alloctor = nullptr, base::DeviceAllocType type = base::DeviceAllocType::UNKNOWN);
    ~Buffer();
    bool create();
    size_t size() const;

private:
    void* buffer_ptr_;
    size_t size_;
    std::shared_ptr<DeviceAlloc> allocator_;
    base::DeviceAllocType device_type_;
};
#endif 