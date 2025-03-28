#ifndef INFER_ALLOC_CPU_H
#define INFER_ALLOC_CPU_H

#include "alloc.h"
#include <memory>
class CpuAlloc: public DeviceAlloc{
public:
    CpuAlloc():DeviceAlloc(base::DeviceType::CPU){}
    ~CpuAlloc() = default;
    void* allocate(size_t size) const override;
    void deallocate(void* ptr) const override;
};


class CpuAllocFactory{
public:
    static std::shared_ptr<CpuAlloc>& get_instance(){
        static std::shared_ptr<CpuAlloc> cpu_allocator_ = std::make_shared<CpuAlloc>();
        return cpu_allocator_;
    }
};
#endif