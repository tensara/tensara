#include <cuda_runtime.h>

__global__ void avg_pool_3d_kernel(const float* input, float* output,
                                    size_t kernel_size, size_t stride, size_t padding,
                                    size_t H, size_t W, size_t D,
                                    size_t out_H, size_t out_W, size_t out_D) {
    size_t out_d = blockIdx.x * blockDim.x + threadIdx.x;
    size_t out_w = blockIdx.y * blockDim.y + threadIdx.y;
    size_t out_h = blockIdx.z;
    if (out_h >= out_H || out_w >= out_W || out_d >= out_D) return;

    float sum = 0.0f;
    float denom = (float)(kernel_size * kernel_size * kernel_size);

    for (size_t m = 0; m < kernel_size; m++) {
        for (size_t n = 0; n < kernel_size; n++) {
            for (size_t o = 0; o < kernel_size; o++) {
                long long r = (long long)(out_h * stride + m) - (long long)padding;
                long long c = (long long)(out_w * stride + n) - (long long)padding;
                long long d = (long long)(out_d * stride + o) - (long long)padding;
                if (r >= 0 && r < (long long)H &&
                    c >= 0 && c < (long long)W &&
                    d >= 0 && d < (long long)D)
                    sum += input[r * W * D + c * D + d];
            }
        }
    }
    output[out_h * out_W * out_D + out_w * out_D + out_d] = sum / denom;
}

extern "C" void solution(const float* input,
                          size_t kernel_size, size_t stride, size_t padding,
                          float* output, size_t H, size_t W, size_t D) {
    size_t out_H = (H + 2 * padding - kernel_size) / stride + 1;
    size_t out_W = (W + 2 * padding - kernel_size) / stride + 1;
    size_t out_D = (D + 2 * padding - kernel_size) / stride + 1;
    dim3 block(8, 8);
    dim3 grid((out_D + 7) / 8, (out_W + 7) / 8, out_H);
    avg_pool_3d_kernel<<<grid, block>>>(input, output, kernel_size, stride, padding,
                                        H, W, D, out_H, out_W, out_D);
    cudaDeviceSynchronize();
}
