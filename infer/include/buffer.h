#ifndef BUFFER_H
#define BUFFER_H


#include "alloc.h"
#include <memory>

class Buffer: public base::NoCopyable, std::enable_shared_from_this<Buffer>{
public:
    explicit Buffer() = default;
    explicit Buffer(size_t size, std::shared_ptr<DeviceAlloc> alloctor = nullptr, base::DeviceType type = base::DeviceType::UNKNOWN);
    ~Buffer();
    bool create();
    size_t size() const;

    base::DeviceType device_type() const{
        return device_type_;
    }

    void* data(){
        return buffer_ptr_;
    }
private:
    void* buffer_ptr_{nullptr};
    size_t size_{0};
    std::shared_ptr<DeviceAlloc> allocator_;
    base::DeviceType device_type_{base::DeviceType::UNKNOWN};
};

#endif 