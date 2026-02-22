#include <cuda_runtime.h>

__global__ void avg_pool_1d_kernel(const float* input, float* output,
                                    size_t kernel_size, size_t stride,
                                    size_t padding, size_t H) {
    size_t H_out = (H + 2 * padding - kernel_size) / stride + 1;
    size_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= H_out) return;

    float sum = 0.0f;
    float denom = (float)kernel_size;
    for (size_t k = 0; k < kernel_size; k++) {
        long long pos = (long long)(out_idx * stride + k) - (long long)padding;
        if (pos >= 0 && pos < (long long)H)
            sum += input[pos];
    }
    output[out_idx] = sum / denom;
}

extern "C" void solution(const float* input,
                          size_t kernel_size, size_t stride, size_t padding,
                          float* output, size_t H) {
    size_t H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int block = 256;
    int grid = (H_out + block - 1) / block;
    avg_pool_1d_kernel<<<grid, block>>>(input, output, kernel_size, stride, padding, H);
    cudaDeviceSynchronize();
}
