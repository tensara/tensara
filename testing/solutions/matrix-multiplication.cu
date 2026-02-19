#include <cuda_runtime.h>

__global__ void matmul_kernel(const float* A, const float* B, float* C,
                               int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++)
            sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

extern "C" void solution(const float* A, const float* B, float* C,
                          int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    matmul_kernel<<<grid, block>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
