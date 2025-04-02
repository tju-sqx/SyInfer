#include "rope_kernel_cuda.cuh"
#include "tensor.h"
#include <cstdint>
#include <cstdio>
#include <cub/block/block_reduce.cuh>
namespace kernel {
    __device__ void rope_calc_fp32(float* input_ptr, float sin_value, float cos_value) {
        float2* input_2_ptr = reinterpret_cast<float2*>(input_ptr);
        *(input_2_ptr) = make_float2(input_2_ptr->x * cos_value - input_2_ptr->y * sin_value, 
        input_2_ptr->x * sin_value + input_2_ptr->y * cos_value);
    }

    __global__ void rope_kernel_fp32(int pos, size_t dim_size, size_t k_dim, size_t head_size, float* key_ptr, float* query_ptr, float* sin_cache_ptr, float* cos_cache_ptr) {
        int id = threadIdx.x + blockDim.x * blockIdx.x;
        id = id * 2;
        if(id >= dim_size) {
            return;
        }
        int head_dim = (id % head_size);

        float sin_value = *(sin_cache_ptr + pos * head_size + head_dim);
        float cos_value = *(cos_cache_ptr + pos * head_size + head_dim);

        rope_calc_fp32(query_ptr + id, sin_value, cos_value);
        if (id >= k_dim) {
            return;
        }
        rope_calc_fp32(key_ptr + id, sin_value, cos_value);

    }
    __global__ void sin_cos_cal_fp32(size_t head_size, size_t max_len_size, float* sin_cache_ptr, float* cos_cache_ptr) {
        int id = threadIdx.x + blockDim.x * blockIdx.x;
        int head_dim = id % head_size;

        for (int pos = 0; pos < max_len_size; ++pos) {
            float frec = 1.0f / pow(10000.0f, static_cast<float>(head_dim) / static_cast<float>(head_size));
            float rotate = static_cast<float>(pos) * frec;
            float sin_value = sinf(rotate);
            float cos_value = cosf(rotate);
            *(sin_cache_ptr + pos * head_size + head_dim) = sin_value;
            *(cos_cache_ptr + pos * head_size + head_dim) = cos_value;
        }
    }
    void sin_cos_calc_cuda(size_t head_size, size_t max_len_size, const Tensor& sin_tensor, const Tensor& cos_tensor, cudaStream_t stream) {
        CHECK(!cos_tensor.empty() && !sin_tensor.empty()) << "Cos OR Sin tensor is empty in sic_cos_cal_cuda func\n";
        float* sin_cache_ptr = const_cast<float*>(sin_tensor.data<float>());
        float* cos_cache_ptr = const_cast<float*>(cos_tensor.data<float>());
        const int THREADS = head_size;
        if (stream) {
            sin_cos_cal_fp32<<<1, THREADS, 0, stream>>>(head_size, max_len_size, sin_cache_ptr, cos_cache_ptr);
        } else {
            sin_cos_cal_fp32<<<1, THREADS>>>(head_size, max_len_size, sin_cache_ptr, cos_cache_ptr);
        }
    }
    void rope_kernel_cuda(size_t dim_size, size_t k_dim, size_t head_size, const Tensor& pos_tensor, const Tensor& key_tensor, const Tensor& query_tensor, const Tensor& sin_tensor, const Tensor& cos_tensor, void* stream) {
        CHECK(!key_tensor.empty()) << "Key tensor is empty in rope_kernel_cuda func\n";
        CHECK(!query_tensor.empty()) << "Query tensor is empty in rope_kernel_cuda func\n";
        CHECK(!pos_tensor.empty()) << "Pos tensor is empty in rope_kernel_cuda func\n";
        CHECK(!cos_tensor.empty() && !sin_tensor.empty()) << "Cos OR Sin tensor is empty in sic_cos_cal_cuda func\n";

        int pos = *(pos_tensor.data<int>());
        cudaStream_t m_stream = static_cast<cudaStream_t>(stream);
        float* key_ptr = const_cast<float*>(key_tensor.data<float>());
        float* query_ptr = const_cast<float*>(query_tensor.data<float>());
        float* sin_cache_ptr = const_cast<float*>(sin_tensor.data<float>());
        float* cos_cache_ptr = const_cast<float*>(cos_tensor.data<float>());
        constexpr int THREADS = 128;
        const int BLOCKS = (dim_size + THREADS + 1) / head_size;
        if (m_stream) {
            rope_kernel_fp32<<<BLOCKS, THREADS, 0, m_stream>>>(pos, dim_size, k_dim, head_size, key_ptr, query_ptr, sin_cache_ptr, cos_cache_ptr);
        } else {
            rope_kernel_fp32<<<BLOCKS, THREADS>>>(pos, dim_size, k_dim, head_size, key_ptr, query_ptr, sin_cache_ptr, cos_cache_ptr);
        }
    }
}