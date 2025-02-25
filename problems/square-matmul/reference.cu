#include <cuda_runtime.h>

__global__ void reference_matrix_multiply(float* A, float* B, float* C, size_t N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

extern "C" void reference_solution(float* input_a, float* input_b, float* output_c, size_t n) {
    dim3 blockDim(16, 16);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, 
                 (n + blockDim.y - 1) / blockDim.y);

    reference_matrix_multiply<<<gridDim, blockDim>>>(input_a, input_b, output_c, n);   
}