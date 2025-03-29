#include "alloc.h"
#include "base.h"
#include <cstring>
#include <cuda_runtime_api.h>
#include <glog/logging.h>

void DeviceAlloc::smemcpy(const void* src_ptr, void* dest_ptr, size_t byte_size, base::DataTransMode mode, void* stream, bool need_sync) {
    CHECK_NE(src_ptr, nullptr);
    CHECK_NE(dest_ptr, nullptr);

    cudaStream_t m_stream = nullptr;
    m_stream = stream == nullptr? nullptr : static_cast<CUstream_st*>(stream);
    if (mode == base::DataTransMode::CPU2CPU) {
        memcpy(dest_ptr, src_ptr, byte_size);
    } else if (mode == base::DataTransMode::CPU2CUDA) {
        if (!stream) {
            cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyHostToDevice);
        } else {
            cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyHostToDevice, m_stream);
        }
    } else if (mode == base::DataTransMode::CUDA2CPU) {
        if (!stream) {
            cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToHost);
        } else {
            cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToHost, m_stream);
        }
    } else if (mode == base::DataTransMode::CUDA2CUDA) {
        if (!stream) {
            cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToDevice);
        } else {
            cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToDevice, m_stream);
        }       
    }

    if (need_sync) {
        cudaDeviceSynchronize();
    }
}   