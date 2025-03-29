#include "buffer.h"
#include "base.h"
#include <iostream>

Buffer::Buffer(size_t byte_size, std::shared_ptr<DeviceAlloc> allocator, 
base::DeviceType type, bool use_external): allocator_(allocator), device_type_(type), byte_size_(byte_size), use_external_(use_external){

}

Buffer::~Buffer(){
    if(buffer_ptr_ && allocator_ && !use_external_) {
        allocator_->deallocate(buffer_ptr_);
        buffer_ptr_ = nullptr;
    }
}

bool Buffer::create(){
    if (allocator_ == nullptr || byte_size_ == 0) {
        return false;  
    }
    if (base::DeviceType::UNKNOWN == device_type_) {
        return false;
    }
    if (use_external_ && buffer_ptr_) {
        return true;
    }
    if (base::DeviceType::CPU == device_type_ || base::DeviceType::GPU == device_type_) {
        buffer_ptr_ = allocator_->allocate(byte_size_);
        return true;
    }
    return false;
}

size_t Buffer::byte_size() const{
    return byte_size_;
}

bool Buffer::init_from_external(void* ptr) {
    if(!use_external_ || !ptr) {
        return false;
    }
    buffer_ptr_ = ptr;
    return true;
}