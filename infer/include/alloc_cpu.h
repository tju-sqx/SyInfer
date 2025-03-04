#ifndef INFER_ALLOC_CPU_H
#define INFER_ALLOC_CPU_H

#include <memory>
#include "alloc.h"

class CpuAlloc: public DeviceAlloc{
public:
    CpuAlloc():DeviceAlloc(base::DeviceType::CPU){}
    ~CpuAlloc() = default;
    void* allocate(size_t size) override;
    void deallocate(void* ptr) override;
};


class CpuAllocFactory{
public:
    static std::shared_ptr<CpuAlloc> get_instance(){
        if(!cpu_allocator_) {
            cpu_allocator_ = std::make_shared<CpuAlloc>();
        }
        return cpu_allocator_;
    }
private:
    static std::shared_ptr<CpuAlloc> cpu_allocator_;
};
#endif