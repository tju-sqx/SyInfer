#include "tensor.h"
#include "alloc_cpu.h"
#include "alloc_gpu.h"
#include "base.h"
#include "buffer.h"
#include "glog/logging.h"
#include <algorithm>
#include <iostream>
#include <memory>
Tensor::Tensor(size_t dim0, base::DateType data_type, std::shared_ptr<DeviceAlloc> allocator, void* external_ptr, bool need_alloc): data_type_(data_type), external_ptr_(external_ptr) {
    dims_.push_back(dim0);
    size_ = cal_size();
    bytes_size_ = size_ * data_type_size(data_type_);
    buffer_ptr_ = std::make_shared<Buffer>(bytes_size_, allocator, allocator? allocator->device_type() : base::DeviceType::UNKNOWN, external_ptr_ != nullptr? true : false);
    if(need_alloc && !external_ptr && allocator) {
        this->create();
    } else if(external_ptr_) {
        buffer_ptr_->init_from_external(external_ptr_);
    }
}

Tensor::Tensor(size_t dim0, size_t dim1, base::DateType data_type, std::shared_ptr<DeviceAlloc> allocator, void* external_ptr, bool need_alloc): data_type_(data_type), external_ptr_(external_ptr){
    dims_.push_back(dim0);
    dims_.push_back(dim1);
    size_ = cal_size();
    bytes_size_ = size_ * data_type_size(data_type_);
    buffer_ptr_ = std::make_shared<Buffer>(bytes_size_, allocator, allocator? allocator->device_type() : base::DeviceType::UNKNOWN, external_ptr_? true : false);
    if(need_alloc && !external_ptr && allocator) {
        this->create();
    } else if(external_ptr_) {
        buffer_ptr_->init_from_external(external_ptr_);
    }
}   
//TODO create逻辑修正
bool Tensor::create(){
    if (!buffer_ptr_) {
        return false;
    }           
    if (external_ptr_) {
        bool flag = buffer_ptr_->init_from_external(external_ptr_);
        //std::cout<<"init from external end()\n";
        return flag;
    }
    return buffer_ptr_->create();
}

size_t Tensor::get_dim(size_t idx) const{
    CHECK(idx < dims_.size()) << "get_dim idx is invalid \n";
    return dims_[idx];
}

bool Tensor::empty() const {
    if(!buffer_ptr_) {
        return true;
    }
    return bytes_size_ == 0;
}

size_t Tensor::cal_size() {
    size_t ans = 1;
    for (auto& dim: dims_) {
        ans *= dim;
    }
    return ans;
}

size_t Tensor::dim() const{
    return dims_.size();
}

base::DeviceType Tensor::device_type() const {
    if(!buffer_ptr_) {
        return base::DeviceType::UNKNOWN;
    }
    return buffer_ptr_->device_type();
}

bool Tensor::is_from_external() const {
    return external_ptr_ != nullptr;
}

bool Tensor::set_external_data(void* external_ptr) {
    if(buffer_ptr_) {
        buffer_ptr_->set_use_external();
        return buffer_ptr_->init_from_external(external_ptr);
    }
    return false;
}

void Tensor::to_cuda(cudaStream_t stream) {
    CHECK_NE(buffer_ptr_, nullptr) << "To_cuda failed with invalid buffer_ptr\n";
    if (buffer_ptr_->device_type() == base::DeviceType::UNKNOWN) {
        LOG(ERROR) << "Tensor has no type in to_cuda func\n";
    } else if (buffer_ptr_->device_type() == base::DeviceType::GPU) {
        LOG(INFO) << "Tensor has been in GPU\n";
    } else {
        auto gpu_allocator= GpuAllocFactory::get_instance();
        std::shared_ptr<Buffer> cu_buffer_ptr = std::make_shared<Buffer>(bytes_size_, gpu_allocator, gpu_allocator->device_type());
        cu_buffer_ptr->create();
        gpu_allocator->smemcpy(const_cast<void*>(buffer_ptr_->data()), cu_buffer_ptr->data(), bytes_size_, base::DataTransMode::CPU2CUDA, stream); 
        buffer_ptr_ = cu_buffer_ptr;
    }
}

void Tensor::to_cpu() {
    CHECK_NE(buffer_ptr_, nullptr) << "To_cpu failed with invalid buffer_ptr\n";
    if (buffer_ptr_->device_type() == base::DeviceType::UNKNOWN) {
        LOG(ERROR) << "Tensor has no type in to_cpu func\n";
    } else if (buffer_ptr_->device_type() == base::DeviceType::CPU) {
        LOG(INFO) << "Tensor has been in CPU\n";
    } else {
        auto cpu_allocator= CpuAllocFactory::get_instance();
        std::shared_ptr<Buffer> cpu_buffer_ptr = std::make_shared<Buffer>(bytes_size_, cpu_allocator, cpu_allocator->device_type());
        cpu_buffer_ptr->create();
        cpu_allocator->smemcpy(buffer_ptr_->data(), cpu_buffer_ptr->data(), bytes_size_, base::DataTransMode::CUDA2CPU);
        buffer_ptr_ = cpu_buffer_ptr;
    }
}

void clone(){
    
}