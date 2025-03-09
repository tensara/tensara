#include <cuda_runtime.h>

__global__ void reference_leaky_relu_kernel(const float* input, float* output, size_t n, size_t m, float alpha) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < n && idy < m) {
        size_t index = idy * n + idx;
        float val = input[index];
        output[index] = val > 0.0f ? val : alpha * val;
    }
}

extern "C" void reference_solution(const float* input, float* output, size_t n, size_t m, float alpha) {
    
    dim3 block_size(16, 16);
    dim3 num_blocks((n + block_size.x - 1) / block_size.x,
                    (m + block_size.y - 1) / block_size.y);
    
    reference_leaky_relu_kernel<<<num_blocks, block_size>>>(input, output, n, m, alpha);
    cudaDeviceSynchronize();
}