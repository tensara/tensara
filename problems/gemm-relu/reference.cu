#include <cuda_runtime.h>

__global__ void reference_gemm_relu_kernel(const float* A, const float* W, const float* b, float* C, 
                                         size_t B, size_t N, size_t M) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < B && col < M) {
        float sum = 0.0f;
        for (size_t k = 0; k < N; k++) {
            sum += A[row * N + k] * W[col * N + k];
        }
        sum += b[col];
        C[row * M + col] = sum > 0.0f ? sum : 0.0f;
    }
}

extern "C" void reference_solution(const float* A, const float* W, const float* b, float* C,
                                 size_t B, size_t N, size_t M) {
    dim3 block_size(16, 16);
    dim3 num_blocks((M + block_size.x - 1) / block_size.x,
                    (B + block_size.y - 1) / block_size.y);

    reference_gemm_relu_kernel<<<num_blocks, block_size>>>(A, W, b, C, B, N, M);
    cudaDeviceSynchronize();
}