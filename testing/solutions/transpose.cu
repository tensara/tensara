#include <cuda_runtime.h>

__global__ void transpose_kernel(const float* A, float* C, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N)
        C[col * M + row] = A[row * N + col];
}

extern "C" void solution(const float* A, float* C, int M, int N) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    transpose_kernel<<<grid, block>>>(A, C, M, N);
    cudaDeviceSynchronize();
}
