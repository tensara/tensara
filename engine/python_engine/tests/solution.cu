#include <cuda_runtime.h>

__global__ void reference_conv1d_kernel(float* A, float* B, float* C, size_t N, size_t K) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        float sum = 0.0f;
        int offset = (K - 1) / 2;
        
        for (size_t j = 0; j < K; j++) {
            int input_idx = idx + j - offset;
            float input_val = 0.0f;
            
            if (input_idx >= 0 && input_idx < N) {
                input_val = A[input_idx];
            }
            
            sum += input_val * B[j];
        }
        
        C[idx] = sum;
    }
}

extern "C" void reference_solution(float* A, float* B, float* C, size_t N, size_t K) {
    int block_size = 256;
    int num_blocks = (N + block_size - 1) / block_size;
    
    reference_conv1d_kernel<<<num_blocks, block_size>>>(A, B, C, N, K);
    cudaDeviceSynchronize();
}