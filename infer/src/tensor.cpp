#include "tensor.h"

Tensor::Tensor(size_t dim0, base::DateType data_type, std::shared_ptr<DeviceAlloc> allocator): data_type_(data_type) {
    dims_.push_back(dim0);
    size_ = cal_size();
    bytes_size_ = size_ * data_type_size(data_type_);
    buffer_ptr_ = std::make_shared<Buffer>(bytes_size_, allocator, allocator->device_type());
}   

bool Tensor::create(){
    if(!buffer_ptr_ ) {
        return false;
    }
    return buffer_ptr_->create();
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