#ifndef ALLOC_GPU_H
#define ALLOC_GPU_H

#include "alloc.h"
#include <memory>
#include <unordered_map>
#include <vector>

struct CudaMem {
    void* data;
    size_t byte_size;
    bool busy;
    
    CudaMem() = default;
    CudaMem(void* ptr, size_t byte_size, bool busy): data(ptr), byte_size(byte_size), busy(busy){}
};

class GpuAlloc: public DeviceAlloc{
public:
    GpuAlloc():DeviceAlloc(base::DeviceType::GPU){}
    ~GpuAlloc() = default;
    void* allocate(size_t size) const override;
    void deallocate(void* ptr) const override;
private:
    mutable std::unordered_map<int, size_t> not_free_mems_size_;
    mutable std::unordered_map<int, std::vector<CudaMem>> common_mems_;
    mutable std::unordered_map<int, std::vector<CudaMem>> big_mems_;
};


class GpuAllocFactory{
public:
    static std::shared_ptr<GpuAlloc>& get_instance(){
        static std::shared_ptr<GpuAlloc> gpu_allocator_ = std::make_shared<GpuAlloc>();
        return gpu_allocator_; 
    }
};
#endif
// ALLOC_GPU_H