#include <cuda_runtime.h>

__global__ void upper_trig_matmul_kernel(const float* A, const float* B,
                                          float* C, size_t N) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col >= N || row > col) return;
    float sum = 0.0f;
    for (size_t k = row; k <= col; k++)
        sum += A[row * N + k] * B[k * N + col];
    C[row * N + col] = sum;
}

extern "C" void solution(const float* A, const float* B, float* C, size_t N) {
    cudaMemset(C, 0, N * N * sizeof(float));
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (N + 15) / 16);
    upper_trig_matmul_kernel<<<grid, block>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
