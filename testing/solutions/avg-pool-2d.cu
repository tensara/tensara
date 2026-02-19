#include <cuda_runtime.h>

__global__ void avg_pool_2d_kernel(const float* input, float* output,
                                    size_t H, size_t W, size_t kernel_size,
                                    size_t stride, size_t padding,
                                    size_t out_H, size_t out_W) {
    size_t out_col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t out_row = blockIdx.y * blockDim.y + threadIdx.y;
    if (out_row >= out_H || out_col >= out_W) return;

    float sum = 0.0f;
    float denom = (float)(kernel_size * kernel_size);

    for (size_t m = 0; m < kernel_size; m++) {
        for (size_t n = 0; n < kernel_size; n++) {
            long long r = (long long)(out_row * stride + m) - (long long)padding;
            long long c = (long long)(out_col * stride + n) - (long long)padding;
            if (r >= 0 && r < (long long)H && c >= 0 && c < (long long)W)
                sum += input[r * W + c];
        }
    }
    output[out_row * out_W + out_col] = sum / denom;
}

extern "C" void solution(const float* input, float* output,
                          size_t H, size_t W, size_t kernel_size,
                          size_t stride, size_t padding) {
    size_t out_H = (H + 2 * padding - kernel_size) / stride + 1;
    size_t out_W = (W + 2 * padding - kernel_size) / stride + 1;
    dim3 block(16, 16);
    dim3 grid((out_W + 15) / 16, (out_H + 15) / 16);
    avg_pool_2d_kernel<<<grid, block>>>(input, output, H, W, kernel_size,
                                        stride, padding, out_H, out_W);
    cudaDeviceSynchronize();
}
