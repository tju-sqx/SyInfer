#include "gemv_kernel_cuda.cuh"
#include <cstddef>


namespace kernel {
template<size_t THREAD_PER_BLOCK, size_t ROW_PRE_BLOCK>
__global__ void matmul_kernel_cuda_fp32(const float* input_ptr, const float* weight_ptr, float* output_ptr, size_t ROW, size_t COL) {
        __shared__ float dot_mul_res[THREAD_PER_BLOCK];
        size_t tid = threadIdx.x;

        size_t start_row = blockIdx.x * THREAD_PER_BLOCK;
        size_t end_row = start_row + ROW_PRE_BLOCK;
        if(start_row >= ROW) {
            return;
        }

        int pack_size = 4;
        int pack_num = COL / pack_size;
        int pack_offset = pack_size * pack_num;
    #pragma unroll
        for (size_t r = start_row; r < end_row; ++r) {
            dot_mul_res[tid] = 0;
            float4* input_4_ptr = (float4*)input_ptr;
            float4* weight_4_ptr = (float4*)weight_ptr + r * COL;

            #pragma unroll
            for(size_t p = tid; p < pack_num; p += blockDim.x) {
                float4* cur_input_ptr = input_4_ptr + p;
                float4* cur_weight_ptr  = weight_4_ptr + p;

                float part_sum = cur_input_ptr.x * cur_weight_ptr.x + cur_input_ptr.y + cur_weight_ptr.y +
                                    cur_input_ptr.z * cur_weight_ptr.z + cur_input_ptr.w + cur_weight_ptr.w;
                dot_mul_res[tid] += part_sum;
            }

            for(size_t p = pack_offset + tid; p < COL; p += blockDim.x) {
                dot_mul_res[tid] += input_ptr[p] * weight_ptr[r * COL + p];
            }

            __syncthreads();

            using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
            __share__ typename BlockReduce::TempStorage temp;
            float sum = BlockReduce(temp).Sum(dot_mul_res[tid]);

            __syncthreads();

            if (tid == 0) {
                output_ptr[r] = sum;
            }

            __syncthreads();
        }
    }

    void gemv_kernel_cuda(const Tensor& input_tensor, const Tensor& weight_tensor, const Tensor& output_tensor, float scale = 1.0f) {
        CHECK(!input_tensor.empty() && input_tensor.dim() <= 2) << "Input_tensor is empty OR has invalid dims in gemv\n ";
        CHECK(input_tensor.device_type() == base::DeviceType::GPU) << "Input tensor runs with wrong device in gemv!\n";

        CHECK(!weight_tensor.empty() && weight_tensor.dim() == 2) << "weight_tensor is empty OR has invalid dims\n";
        CHECK(weight_tensor.device_type() == base::DeviceType::GPU) << "weight_tensor runs with wrong device in gemv!\n";

        const size_t ROW = weight_tensor.get_dim(0);
        const size_t COL = weight_tensor.get_dim(1);

        CHECK(ROW == input_tensor.get_dim(0)) << "Input_tensor dismatch weight_tensor in gemv\n";
        matmul_kernel_cuda_fp32<128, 1><<<COL, 128>>>(input_tensor.data<float>(), weight_tensor.data<float>(), const_cast<float*>(output_tensor.data<float>()), ROW, COL);
        
    } 
}