#include <cuda_runtime.h>

__global__ void relu_kernel(const float* A, float* C, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    if (idx < total)
        C[idx] = A[idx] > 0.0f ? A[idx] : 0.0f;
}

extern "C" void solution(const float* A, float* C, int M, int N) {
    int total = M * N;
    int block = 256;
    int grid = (total + block - 1) / block;
    relu_kernel<<<grid, block>>>(A, C, M, N);
    cudaDeviceSynchronize();
}
