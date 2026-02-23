#include <cuda_runtime.h>

__global__ void threshold_kernel(const float* input, float threshold_value,
                                  float* output, size_t height, size_t width) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < height * width)
        output[idx] = input[idx] > threshold_value ? 255.0f : 0.0f;
}

extern "C" void solution(const float* input_image, float threshold_value,
                          float* output_image, size_t height, size_t width) {
    size_t total = height * width;
    int block = 256;
    int grid = (total + block - 1) / block;
    threshold_kernel<<<grid, block>>>(input_image, threshold_value, output_image, height, width);
    cudaDeviceSynchronize();
}
