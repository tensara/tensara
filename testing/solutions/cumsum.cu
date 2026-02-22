#include <cuda_runtime.h>

// Naive sequential scan (correct, not fast)
__global__ void cumsum_kernel(const float* input, float* output, size_t N) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        output[0] = input[0];
        for (size_t i = 1; i < N; i++)
            output[i] = output[i-1] + input[i];
    }
}

extern "C" void solution(const float* input, float* output, size_t N) {
    cumsum_kernel<<<1, 1>>>(input, output, N);
    cudaDeviceSynchronize();
}
