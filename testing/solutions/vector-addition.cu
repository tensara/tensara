#include <cuda_runtime.h>

__global__ void vector_add_kernel(const float* A, const float* B, float* C, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        C[idx] = A[idx] + B[idx];
}

extern "C" void solution(const float* A, const float* B, float* C, size_t N) {
    int block = 256;
    int grid = (N + block - 1) / block;
    vector_add_kernel<<<grid, block>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
