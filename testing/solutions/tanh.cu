#include <cuda_runtime.h>
#include <math.h>

__global__ void tanh_kernel(const float* input, float* output, size_t n, size_t m) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * m)
        output[idx] = tanhf(input[idx]);
}

extern "C" void solution(const float* input, float* output, size_t n, size_t m) {
    size_t total = n * m;
    int block = 256;
    int grid = (total + block - 1) / block;
    tanh_kernel<<<grid, block>>>(input, output, n, m);
    cudaDeviceSynchronize();
}
