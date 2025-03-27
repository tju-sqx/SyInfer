#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include "buffer.h"
#include <glog/logging.h>
class Tensor {
private:
    size_t size_;
    size_t bytes_size_;
    void* external_ptr_{nullptr};

    std::vector<size_t> dims_{};
    std::shared_ptr<Buffer> buffer_ptr_{nullptr};
    base::DateType data_type_{base::DateType::DATA_UNKONWN};
    
    const size_t cal_size();

public:
    explicit Tensor(size_t dim0, base::DateType data_type, std::shared_ptr<DeviceAlloc> alloc_ptr, void* external_ptr = nullptr);
    explicit Tensor(size_t dim0, size_t dim1, base::DateType data_type, std::shared_ptr<DeviceAlloc> allocator, void* external_ptr = nullptr);
    ~Tensor() = default;
    size_t dim() const;
    bool create();
    base::DeviceType device_type() const;
    const bool empty() const;
    const size_t get_dim(size_t idx) const;
    const bool is_from_external() const;
    const bool set_external_data(void* ptr);
    void set_device_type(base::DeviceType type) {
        buffer_ptr_->set_device_type(type);
    }

    template<typename T> 
    const T* data() const;

    template<typename T>
    T* data(); 

    
    const size_t size() const{
        return size_;
    }

    const size_t byte_size() const{
        return bytes_size_;
    }

};

template<typename T>
const T* Tensor::data() const{
    if(!buffer_ptr_){
        return nullptr;
    }
    return reinterpret_cast<T*>(buffer_ptr_->data());
}

template<typename T>
T* Tensor::data(){
    if(!buffer_ptr_){
        return nullptr;
    }
    return reinterpret_cast<T*>(buffer_ptr_->data());
}
#endif