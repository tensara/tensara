#include <cuda_runtime.h>

__global__ void reference_conv2d_kernel(const float* A, const float* B, float* C, size_t H, size_t W, size_t Kh, size_t Kw) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < H && col < W) {
        float sum = 0.0f;
        int offset_h = (Kh - 1) / 2;
        int offset_w = (Kw - 1) / 2;
        
        for (size_t i = 0; i < Kh; i++) {
            for (size_t j = 0; j < Kw; j++) {
                int input_row = row + i - offset_h;
                int input_col = col + j - offset_w;
                float input_val = 0.0f;
                
                if (input_row >= 0 && input_row < H && 
                    input_col >= 0 && input_col < W) {
                    input_val = A[input_row * W + input_col];
                }
                
                sum += input_val * B[i * Kw + j];
            }
        }
        
        C[row * W + col] = sum;
    }
}

extern "C" void reference_solution(const float* A, const float* B, float* C, size_t H, size_t W, size_t Kh, size_t Kw) {
    dim3 block_size(16, 16);
    dim3 num_blocks((W + block_size.x - 1) / block_size.x,
                    (H + block_size.y - 1) / block_size.y);
    
    reference_conv2d_kernel<<<num_blocks, block_size>>>(A, B, C, H, W, Kh, Kw);
    cudaDeviceSynchronize();
}