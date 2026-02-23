#include <cuda_runtime.h>

__global__ void conv2d_kernel(const float* A, const float* B, float* C,
                               size_t H, size_t W, size_t Kh, size_t Kw) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= H || col >= W) return;

    float sum = 0.0f;
    long long rh = (long long)(Kh / 2);
    long long rw = (long long)(Kw / 2);
    for (size_t m = 0; m < Kh; m++) {
        for (size_t n = 0; n < Kw; n++) {
            long long r = (long long)row + (long long)m - rh;
            long long c = (long long)col + (long long)n - rw;
            if (r >= 0 && r < (long long)H && c >= 0 && c < (long long)W)
                sum += A[r * W + c] * B[m * Kw + n];
        }
    }
    C[row * W + col] = sum;
}

extern "C" void solution(const float* A, const float* B, float* C,
                          size_t H, size_t W, size_t Kh, size_t Kw) {
    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);
    conv2d_kernel<<<grid, block>>>(A, B, C, H, W, Kh, Kw);
    cudaDeviceSynchronize();
}
