#include <cstddef>
#include <cub/cub.cuh>
#include <cstdint>
#include <endian.h>

#include "mha_kernel_cuda.cuh"

namespace kernel {
    __device__ void softmax_fp32(float* __restrict__ score, size_t size) {
        int tid = threadIdx.x;
        int step = blockDim.x;

        float max_value = tid < size? score[tid] : 0;
        for (int t = tid + step; t < size; t += step) {
            if (score[t] > max_value) {
                max_value = score[t];
            }
        }

        using BlockReduce = cub::BlockReduce<float, 128>;
        __shared__ BlockReduce::TempStorage temp;
        __shared__ float share_value;
        max_value = BlockReduce(temp).Reduce(max_value, cub::Max());

        //different
        if (tid == 0) {
            share_value = max_value;
        }
        __syncthreads();
        max_value = share_value;

        float sum = 0.0f;
        for (int t = tid; t < size; t += step) {
            score[t] = expf(score[t] - max_value);
            sum += score[t];
        }

        sum = BlockReduce(temp).Sum(sum);
        if (tid == 0) {
            share_value = sum;
        }
        __syncthreads();
        sum = share_value;
        for (int t = tid; t < size; t += step) {
            score[t] /= sum;
        }
    }

    __global__ void mha_fp32(size_t pos, size_t seq_len, size_t head_num, size_t head_size, size_t kv_mul, size_t layer_offset, size_t kv_dim, float* out_ptr, float* query_ptr ,float* key_ptr, float* value_ptr, float* score_ptr) {
        float scale = 1.0f / sqrtf(head_size);

        int h = blockIdx.x;
        if (h >= head_num) {
            return;
        }

        size_t cache_offset = (h / kv_mul) * head_size;
        float* cur_query_ptr = query_ptr + head_size * h;
        float* cur_score_ptr = score_ptr + seq_len * h;
        for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
            float* cur_key_ptr = key_ptr + layer_offset + cache_offset + t * kv_dim;
            float score = 0.0f;
        #pragma unroll
            for (int i = 0; i < head_size; i += 4) {
                float4* cur_query_4_ptr = reinterpret_cast<float4*>(cur_query_ptr + i);
                float4* cur_key_4_ptr = reinterpret_cast<float4*>(cur_key_ptr + i);
                if (i < head_size) {
                    score += cur_query_4_ptr->x * cur_key_4_ptr->x;
                }
                if (i + 1 < head_size) {
                    score += cur_query_4_ptr->y * cur_key_4_ptr->y;
                }
                if (i + 2 < head_size) {
                    score += cur_query_4_ptr->z * cur_key_4_ptr->z;
                }
                if (i + 3 < head_size) {
                    score += cur_query_4_ptr->w * cur_key_4_ptr->w;
                }
            }

            score *= scale;
            cur_score_ptr[t] = score;
            
        }
        __syncthreads();
        softmax_fp32(cur_score_ptr, pos + 1);
        __syncthreads();

       float* cur_output_ptr = out_ptr + h * head_size;
        for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
            float output = 0.0f;
            #pragma unroll
            for(int t = 0; t <= pos; ++t) {
                float* cur_value_ptr = value_ptr + layer_offset + (h /kv_mul) * head_size + t * kv_dim;
                float score = cur_score_ptr[t] * cur_value_ptr[i];
                output += score;
            }
            cur_output_ptr[i] = output;
        }

    }

    void mha_cuda_kernel(size_t pos, size_t seq_len, size_t kv_dim, size_t head_num, size_t head_size, size_t kv_mul, size_t layer_idx,
    const Tensor& mha_out, const Tensor& query_tensor, const Tensor& key_tensor, const Tensor& value_tensor, const Tensor& score_tensor, base::DeviceType device_type) {

        float* out_ptr = const_cast<float*>(mha_out.data<float>());
        float* query_ptr = const_cast<float*>(query_tensor.data<float>());
        float* key_ptr = const_cast<float*>(key_tensor.data<float>());
        float* value_ptr = const_cast<float*>(value_tensor.data<float>());
        float* score_ptr = const_cast<float*>(score_tensor.data<float>());

        size_t layer_offset = layer_idx * kv_dim * seq_len;

        mha_fp32<<<head_num, 128, 0>>>(pos, seq_len, head_num, head_size,kv_mul, layer_offset, kv_dim,
        out_ptr, query_ptr, key_ptr, value_ptr, score_ptr);
    }
}