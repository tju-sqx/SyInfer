#include "buffer.h"
#include <iostream>

Buffer::Buffer(size_t size, std::shared_ptr<DeviceAlloc> allocator, 
base::DeviceType type): allocator_(allocator), device_type_(type), size_(size){
    
}

Buffer::~Buffer(){
    if(buffer_ptr_ && allocator_) {
        allocator_->deallocate(buffer_ptr_);
        buffer_ptr_ = nullptr;
    }
}

bool Buffer::create(){
    if(allocator_ == nullptr || size_ == 0){
        return false;  
    }
    if(base::DeviceType::UNKNOWN == device_type_){
        return false;
    }

    if(base::DeviceType::CPU == device_type_) {
        buffer_ptr_ = allocator_->allocate(size_);
        return true;
    }
    return false;
}

size_t Buffer::size() const{
    return size_;
}