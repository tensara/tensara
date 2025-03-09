#include <cuda_runtime.h>

__global__ void reference_vector_add(const float* input1, const float* input2, float* output, size_t n) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input1[idx] + input2[idx];
    }
}

extern "C" void reference_solution(const float* d_input1, const float* d_input2, float* d_output, size_t n) {
    const int block_size = 256;
    const int num_blocks = (n + block_size - 1) / block_size;
    reference_vector_add<<<num_blocks, block_size>>>(d_input1, d_input2, d_output, n);
} 