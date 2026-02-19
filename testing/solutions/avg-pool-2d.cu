#include <cuda_runtime.h>

__global__ void avg_pool_2d_kernel(const float* input, float* output,
                                    int H, int W, int kernel_size,
                                    int stride, int padding,
                                    int out_H, int out_W) {
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    if (out_row >= out_H || out_col >= out_W) return;

    float sum = 0.0f;
    int count = kernel_size * kernel_size;
    for (int m = 0; m < kernel_size; m++) {
        for (int n = 0; n < kernel_size; n++) {
            int r = out_row * stride + m - padding;
            int c = out_col * stride + n - padding;
            if (r >= 0 && r < H && c >= 0 && c < W)
                sum += input[r * W + c];
        }
    }
    output[out_row * out_W + out_col] = sum / count;
}

extern "C" void solution(const float* input, float* output,
                          int H, int W, int kernel_size, int stride, int padding) {
    int out_H = (H + 2 * padding - kernel_size) / stride + 1;
    int out_W = (W + 2 * padding - kernel_size) / stride + 1;
    dim3 block(16, 16);
    dim3 grid((out_W + 15) / 16, (out_H + 15) / 16);
    avg_pool_2d_kernel<<<grid, block>>>(input, output, H, W, kernel_size,
                                        stride, padding, out_H, out_W);
    cudaDeviceSynchronize();
}
