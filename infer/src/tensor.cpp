#include "tensor.h"
#include "glog/logging.h"
Tensor::Tensor(size_t dim0, base::DateType data_type, std::shared_ptr<DeviceAlloc> allocator, void* external_ptr): data_type_(data_type), external_ptr_(external_ptr) {
    dims_.push_back(dim0);
    size_ = cal_size();
    bytes_size_ = size_ * data_type_size(data_type_);
    buffer_ptr_ = std::make_shared<Buffer>(bytes_size_, allocator, allocator->device_type(), external_ptr != nullptr? true : false);
}

Tensor::Tensor(size_t dim0, size_t dim1, base::DateType data_type, std::shared_ptr<DeviceAlloc> allocator, void* external_ptr): data_type_(data_type), external_ptr_(external_ptr){
    dims_.push_back(dim0);
    dims_.push_back(dim1);
    size_ = cal_size();
    bytes_size_ = size_ * data_type_size(data_type_);
    buffer_ptr_ = std::make_shared<Buffer>(bytes_size_, allocator, allocator->device_type(), external_ptr != nullptr? true : false);
}   

bool Tensor::create(){
    if (!buffer_ptr_) {
        return false;
    }           
    if (external_ptr_) {
        return buffer_ptr_->init_from_external(external_ptr_);
    }
    return buffer_ptr_->create();
}

const size_t Tensor::get_dim(size_t idx) const{
    CHECK(idx < dims_.size()) << "get_dim idx is invalid \n";
    return dims_[idx];
}

const bool Tensor::empty() const {
    if(!buffer_ptr_) {
        return true;
    }
    return bytes_size_ == 0;
}

const size_t Tensor::cal_size() {
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

const bool Tensor::is_from_external() const {
    return external_ptr_ != nullptr;
}

const bool Tensor::set_external_data(void* external_ptr) {
    if(buffer_ptr_) {
        buffer_ptr_->set_use_external();
        return buffer_ptr_->init_from_external(external_ptr);
    }
    return false;
}