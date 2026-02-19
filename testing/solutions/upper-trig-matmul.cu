#include <cuda_runtime.h>

__global__ void upper_trig_matmul_kernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col >= N || row > col) return;  // only upper triangle

    float sum = 0.0f;
    // A[row][k] is 0 for k < row, B[k][col] is 0 for k > col
    for (int k = row; k <= col; k++)
        sum += A[row * N + k] * B[k * N + col];
    C[row * N + col] = sum;
}

extern "C" void solution(const float* A, const float* B, float* C, int N) {
    // Zero out C first since we only write upper triangle
    cudaMemset(C, 0, N * N * sizeof(float));
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (N + 15) / 16);
    upper_trig_matmul_kernel<<<grid, block>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
