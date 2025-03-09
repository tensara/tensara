#include <cuda_runtime.h>

__global__ void reference_matrix_multiply(const float* A, const float* B, float* C, size_t M, size_t N, size_t K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

extern "C" void reference_solution(const float* input_a, const float* input_b, float* output_c, size_t m, size_t n, size_t k) {
    dim3 blockDim(16, 16);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, 
                 (m + blockDim.y - 1) / blockDim.y);

    reference_matrix_multiply<<<gridDim, blockDim>>>(input_a, input_b, output_c, m, n, k);   
}