#include <cuda_runtime.h>
#include <float.h>

__global__ void max_pool_1d_kernel(const float* input, float* output,
                                    int N, int kernel_size, int stride,
                                    int padding, int dilation, int out_len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= out_len) return;

    float max_val = -FLT_MAX;
    for (int m = 0; m < kernel_size; m++) {
        int pos = i * stride + m * dilation - padding;
        if (pos >= 0 && pos < N)
            max_val = fmaxf(max_val, input[pos]);
    }
    output[i] = max_val;
}

extern "C" void solution(const float* input, float* output,
                          int N, int kernel_size, int stride,
                          int padding, int dilation) {
    int out_len = (N + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int block = 256;
    int grid = (out_len + block - 1) / block;
    max_pool_1d_kernel<<<grid, block>>>(input, output, N, kernel_size, stride,
                                        padding, dilation, out_len);
    cudaDeviceSynchronize();
}
