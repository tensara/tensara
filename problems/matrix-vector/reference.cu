#include <cuda_runtime.h>

__global__ void reference_matrix_vector_multiply(float* A, float* B, float* C, size_t M, size_t K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k];
        }
        C[row] = sum;
    }
}

extern "C" void reference_solution(float* input_a, float* input_b, float* output_c, size_t m, size_t k) {
    const int BLOCK_SIZE = 256;
    
    int gridSize = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    reference_matrix_vector_multiply<<<gridSize, BLOCK_SIZE>>>(input_a, input_b, output_c, m, k);
}