#include <cuda_runtime.h>

__global__ void transpose_kernel(const float* A, float* C, size_t M, size_t N) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N)
        C[col * M + row] = A[row * N + col];
}

extern "C" void solution(const float* A, float* C, size_t M, size_t N) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    transpose_kernel<<<grid, block>>>(A, C, M, N);
    cudaDeviceSynchronize();
}
