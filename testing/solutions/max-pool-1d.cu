#include <cuda_runtime.h>
#include <float.h>

__global__ void max_pool_1d_kernel(const float* input, float* output,
                                    size_t kernel_size, size_t stride,
                                    size_t padding, size_t dilation, size_t H) {
    size_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t H_out = (H + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    if (out_idx >= H_out) return;

    float max_val = -FLT_MAX;
    for (size_t k = 0; k < kernel_size; k++) {
        long long pos = (long long)(out_idx * stride + k * dilation) - (long long)padding;
        if (pos >= 0 && pos < (long long)H)
            max_val = fmaxf(max_val, input[pos]);
    }
    output[out_idx] = max_val;
}

extern "C" void solution(const float* input,
                          size_t kernel_size, size_t stride, size_t padding, size_t dilation,
                          float* output, size_t H) {
    size_t H_out = (H + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int block = 256;
    int grid = (H_out + block - 1) / block;
    max_pool_1d_kernel<<<grid, block>>>(input, output, kernel_size, stride, padding, dilation, H);
    cudaDeviceSynchronize();
}
