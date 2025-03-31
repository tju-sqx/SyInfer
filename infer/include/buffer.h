#ifndef BUFFER_H
#define BUFFER_H


#include "alloc.h"
#include <memory>

class Buffer: public base::NoCopyable, std::enable_shared_from_this<Buffer>{
public:
    explicit Buffer() = default;
    explicit Buffer(size_t byte_size, std::shared_ptr<DeviceAlloc> alloctor = nullptr, base::DeviceType type = base::DeviceType::UNKNOWN, bool use_external = false);
    ~Buffer();
    bool create();
    bool init_from_external(void* ptr);
    size_t byte_size() const;
    bool is_use_external() const {return use_external_;}
    void set_use_external() {use_external_ = true;}
    void set_device_type(base::DeviceType type) {device_type_ = type;}
    base::DeviceType device_type() const{
        return device_type_;
    }

    void* data(){
        return buffer_ptr_;
    }
private:
    void* buffer_ptr_{nullptr};
    size_t byte_size_{0};
    bool use_external_{false};
    std::shared_ptr<DeviceAlloc> allocator_;
    base::DeviceType device_type_{base::DeviceType::UNKNOWN};
};

#endif 