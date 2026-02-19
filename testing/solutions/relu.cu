#include <cuda_runtime.h>

__global__ void relu_kernel(const float* A, float* C, size_t M, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N)
        C[idx] = A[idx] > 0.0f ? A[idx] : 0.0f;
}

extern "C" void solution(const float* A, float* C, size_t M, size_t N) {
    size_t total = M * N;
    int block = 256;
    int grid = (total + block - 1) / block;
    relu_kernel<<<grid, block>>>(A, C, M, N);
    cudaDeviceSynchronize();
}
