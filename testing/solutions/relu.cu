#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, size_t n, size_t m) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * m)
        output[idx] = input[idx] > 0.0f ? input[idx] : 0.0f;
}

extern "C" void solution(const float* input, float* output, size_t n, size_t m) {
    size_t total = n * m;
    int block = 256;
    int grid = (total + block - 1) / block;
    relu_kernel<<<grid, block>>>(input, output, n, m);
    cudaDeviceSynchronize();
}
