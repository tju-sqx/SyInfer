#include "alloc_gpu.h"
#include <cstddef>
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <vector>

void* GpuAlloc::allocate(size_t byte_size) const {
    int id = -1;
    cudaError_t state = cudaGetDevice(&id);
    CHECK(state == cudaSuccess) << "Can not get cuda device id in allocate func\n";
    if(byte_size > 1024 * 1024) {
        auto& cur_big_mems = big_mems_[id];
        int select_id = -1;
        for (size_t i = 0; i < cur_big_mems.size(); ++i) {
            if(cur_big_mems[i].byte_size >= byte_size && !cur_big_mems[i].busy &&
            cur_big_mems[i].byte_size - byte_size < static_cast<size_t>(1 * 1024 * 1024)) {
                if(select_id == -1 || cur_big_mems[i].byte_size < cur_big_mems[select_id].byte_size) {
                    select_id = i;
                }
            }
        }
        if (select_id != -1) {
            cur_big_mems[select_id].busy = true;
            return cur_big_mems[select_id].data;       
        }
        void* data_ptr = nullptr;
        state = cudaMalloc(&data_ptr, byte_size);
        if(state != cudaSuccess) {
            char buf[256];
                snprintf(buf, 256,
                "Error: CUDA error when allocating %lu MB memory! maybe there's no enough memory "
                "left on  device.",
               byte_size >> 20);
            LOG(ERROR) << buf;
            return nullptr;
        }
        cur_big_mems.emplace_back(data_ptr, byte_size, true);
        return data_ptr;
    }
    
    auto& cur_mems = common_mems_[id];
    for (size_t i = 0; i < cur_mems.size(); ++i) {
        if (cur_mems[i].byte_size >= byte_size && !cur_mems[i].busy) {
            cur_mems[i].busy = true;
            not_free_mems_size_[id] -= cur_mems[i].byte_size;
            return cur_mems[i].data;
        }
    }
    void* data_ptr = nullptr;
    state = cudaMalloc(&data_ptr ,byte_size);
    if(state != cudaSuccess) {
        char buf[256];
        snprintf(buf, 256,
"Error: CUDA error when allocating %lu MB memory! maybe there's no enough memory "
        "left on  device.",
        byte_size >> 20);
        LOG(ERROR) << buf;
        return nullptr;
    }
    common_mems_[id].emplace_back(data_ptr, byte_size, true);
    return data_ptr;
}

void GpuAlloc::deallocate(void* ptr) const{
    if (!ptr) {
        return ;
    }

    cudaError_t state = cudaSuccess;
    for (auto& it: common_mems_) {
        if (not_free_mems_size_[it.first] >= static_cast<size_t>(1 * 1024 * 1024 * 1024)) {
            auto& cur_mems = it.second;
            std::vector<CudaMem> tmp;
            for (auto& mem: cur_mems) {
                if(mem.busy == false) {
                    state = cudaSetDevice(it.first);
                    state = cudaFree(mem.data);
                    CHECK(state == cudaSuccess)
                    << "Error: CUDA error when release memory on device " << it.first;
                } else {
                    tmp.emplace_back(mem);
                }
            }
            cur_mems.clear();
            it.second = tmp;
            not_free_mems_size_[it.first] = 0;
        }
    }

    for (auto& it: common_mems_) {
        for (auto& mem: common_mems_[it.first]) {
            if(mem.data == ptr) {
                mem.busy = false;
                not_free_mems_size_[it.first] += mem.byte_size;
                return;
            }
        }
        for (auto& bie_mem: big_mems_[it.first]) {
            if(bie_mem.data == ptr) {
                bie_mem.busy = false;
                return;
            }
        }
    }
    state = cudaFree(ptr);
    CHECK(state == cudaSuccess) << "Error: CUDA error when release memory on device";
}