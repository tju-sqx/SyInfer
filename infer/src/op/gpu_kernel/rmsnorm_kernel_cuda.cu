#include "base.h"
#include "rmsnorm_kernel_cuda.cuh"
#include <cstddef>
#include <cstdio>
#include <cub/block/block_reduce.cuh>

namespace kernel {
    template <size_t THREAD_PRE_BLOCK>
    __global__ void rmsnorm_fp32(size_t size, float eps, float* input_ptr, float* weight_ptr, float* output_ptr) {

        int pack_size = 4;
        int pack_num = size / pack_size;
        int pack_offset = pack_size * pack_size;

        float sum = 0.0f;
        for (int t = threadIdx.x; t < pack_num; t += blockDim.x) {
            float4* input_4_ptr = reinterpret_cast<float4*>(input_ptr + t * pack_size);
            sum += input_4_ptr->x * input_4_ptr->x +
                   input_4_ptr->y * input_4_ptr->y + 
                   input_4_ptr->z * input_4_ptr->z + 
                   input_4_ptr->w * input_4_ptr->w;
        }

        for (int t = pack_offset + threadIdx.x; t < size; t += blockDim.x) {
            sum += input_ptr[t] * input_ptr[t];
        }

        using BlockReduce = cub::BlockReduce<float, THREAD_PRE_BLOCK>;
        __shared__ typename BlockReduce::TempStorage tmp;
        __shared__ float share_val;
        sum = BlockReduce(tmp).Sum(sum);
        if(threadIdx.x == 0) {
            share_val = sum;
        }
        __syncthreads();
        sum = share_val;
        float scale = 1.0f / sqrtf(sum / static_cast<float>(size));
        for (int t = threadIdx.x; t < pack_num; t += blockDim.x) {
            float4* input_4_ptr = reinterpret_cast<float4*>(input_ptr + t * pack_size);
            float4* output_4_ptr = reinterpret_cast<float4*>(output_ptr + t * pack_size);
            float4* weight_4_ptr = reinterpret_cast<float4*>(weight_ptr + t * pack_size);
            *output_4_ptr = make_float4(weight_4_ptr->x * scale * input_4_ptr->x,
                                        weight_4_ptr->y * scale * input_4_ptr->y,
                                        weight_4_ptr->z * scale * input_4_ptr->z,
                                        weight_4_ptr->w * scale * input_4_ptr->w);
        }

        for (int t = pack_offset + threadIdx.x; t < size; t += blockDim.x) {
            output_ptr[t] = input_ptr[t] * scale * weight_ptr[t];
        }
    }
    void rmsnorm_kernel_cuda(const Tensor& input_tensor, const Tensor& weight_tensor, const Tensor& output_tensor, void* stream) {
        CHECK(!input_tensor.empty()) << "Input tensor is empty in rmsnorm_cuda_kernel\n";
        CHECK(!weight_tensor.empty()) << "Weight tensor is empty in rmsnorm_cuda_kernel\n";
        CHECK(!output_tensor.empty()) << "Output tensor is empty in rmsnorm_cuda_kernel\n";
        CHECK(input_tensor.device_type() == base::DeviceType::GPU &&
        weight_tensor.device_type() == base::DeviceType::GPU &&
        output_tensor.device_type() == base::DeviceType::GPU ) << "Tensor is not on GPU in rmsnorm_cuda_kernel\n";
        const size_t size = input_tensor.size();
        float* input_ptr = const_cast<float*>(input_tensor.data<float>());
        float* weight_ptr = const_cast<float*>(weight_tensor.data<float>());
        float* out_ptr = const_cast<float*>(output_tensor.data<float>());
        constexpr size_t THREADS = 128;
        constexpr float eps = 1e-5;
        if(stream) {
            cudaStream_t m_stream = static_cast<cudaStream_t>(stream);
            rmsnorm_fp32<THREADS><<<1, THREADS, 0, m_stream>>>(size, eps, input_ptr, weight_ptr, out_ptr);
        } else {
            rmsnorm_fp32<THREADS><<<1, THREADS >>>(size, eps, input_ptr, weight_ptr, out_ptr);
        }
        
    }
}