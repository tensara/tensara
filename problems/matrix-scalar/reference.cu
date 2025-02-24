#include <cuda_runtime.h>

__global__ void reference_matrix_scalar_multiply(float* A, float B, float* C, size_t N, size_t M) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M) {
        for (int col = 0; col < N; col++) {
            C[row * N + col] = A[row * N + col] * B;
        }
    }
}

extern "C" void reference_solution(float* input_a, float input_scalar, float* output_c, size_t n, size_t m) {
    const int BLOCK_SIZE = 256;
    
    int gridSize = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    reference_matrix_scalar_multiply<<<gridSize, BLOCK_SIZE>>>(input_a, input_scalar, output_c, n, m);
}