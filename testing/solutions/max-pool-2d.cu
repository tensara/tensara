#include <cuda_runtime.h>
#include <float.h>

__global__ void max_pool_2d_kernel(const float* input, float* output,
                                    size_t kernel_size, size_t stride,
                                    size_t padding, size_t dilation,
                                    size_t H, size_t W, size_t out_H, size_t out_W) {
    size_t out_col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t out_row = blockIdx.y * blockDim.y + threadIdx.y;
    if (out_row >= out_H || out_col >= out_W) return;

    float max_val = -FLT_MAX;
    for (size_t m = 0; m < kernel_size; m++) {
        for (size_t n = 0; n < kernel_size; n++) {
            long long r = (long long)(out_row * stride + m * dilation) - (long long)padding;
            long long c = (long long)(out_col * stride + n * dilation) - (long long)padding;
            if (r >= 0 && r < (long long)H && c >= 0 && c < (long long)W)
                max_val = fmaxf(max_val, input[r * W + c]);
        }
    }
    output[out_row * out_W + out_col] = max_val;
}

extern "C" void solution(const float* input,
                          size_t kernel_size, size_t stride, size_t padding, size_t dilation,
                          float* output, size_t H, size_t W) {
    size_t out_H = (H + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    size_t out_W = (W + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    dim3 block(16, 16);
    dim3 grid((out_W + 15) / 16, (out_H + 15) / 16);
    max_pool_2d_kernel<<<grid, block>>>(input, output, kernel_size, stride, padding, dilation, H, W, out_H, out_W);
    cudaDeviceSynchronize();
}
