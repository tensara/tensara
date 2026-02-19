#include <cuda_runtime.h>
#include <float.h>

__global__ void max_pool_1d_kernel(const float* input, float* output,
                                    size_t N, size_t kernel_size, size_t stride,
                                    size_t padding, size_t dilation, size_t out_len) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= out_len) return;

    float max_val = -FLT_MAX;
    for (size_t m = 0; m < kernel_size; m++) {
        long long pos = (long long)(i * stride) + (long long)(m * dilation) - (long long)padding;
        if (pos >= 0 && pos < (long long)N)
            max_val = fmaxf(max_val, input[pos]);
    }
    output[i] = max_val;
}

extern "C" void solution(const float* input, float* output,
                          size_t N, size_t kernel_size, size_t stride,
                          size_t padding, size_t dilation) {
    size_t out_len = (N + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int block = 256;
    int grid = ((int)out_len + block - 1) / block;
    max_pool_1d_kernel<<<grid, block>>>(input, output, N, kernel_size, stride,
                                        padding, dilation, out_len);
    cudaDeviceSynchronize();
}
