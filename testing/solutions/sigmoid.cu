#include <cuda_runtime.h>
#include <math.h>

__global__ void sigmoid_kernel(const float* A, float* C, size_t M, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N)
        C[idx] = 1.0f / (1.0f + expf(-A[idx]));
}

extern "C" void solution(const float* A, float* C, size_t M, size_t N) {
    size_t total = M * N;
    int block = 256;
    int grid = (total + block - 1) / block;
    sigmoid_kernel<<<grid, block>>>(A, C, M, N);
    cudaDeviceSynchronize();
}
