#include <cuda_runtime.h>

__global__ void hard_sigmoid_kernel(const float* A, float* C, size_t M, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) {
        float x = A[idx];
        if (x <= -3.0f)     C[idx] = 0.0f;
        else if (x >= 3.0f) C[idx] = 1.0f;
        else                C[idx] = (x + 3.0f) / 6.0f;
    }
}

extern "C" void solution(const float* A, float* C, size_t M, size_t N) {
    size_t total = M * N;
    int block = 256;
    int grid = (total + block - 1) / block;
    hard_sigmoid_kernel<<<grid, block>>>(A, C, M, N);
    cudaDeviceSynchronize();
}
