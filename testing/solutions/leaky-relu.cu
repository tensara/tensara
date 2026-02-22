#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(const float* input, float alpha, float* output, size_t n, size_t m) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * m) {
        float x = input[idx];
        output[idx] = x > 0.0f ? x : alpha * x;
    }
}

extern "C" void solution(const float* input, float alpha, float* output, size_t n, size_t m) {
    size_t total = n * m;
    int block = 256;
    int grid = (total + block - 1) / block;
    leaky_relu_kernel<<<grid, block>>>(input, alpha, output, n, m);
    cudaDeviceSynchronize();
}
