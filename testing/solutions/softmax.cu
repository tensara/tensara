#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

__global__ void softmax_kernel(const float* A, float* C, size_t M, size_t N) {
    size_t row = blockIdx.x;
    if (row >= M) return;

    const float* row_in  = A + row * N;
    float*       row_out = C + row * N;

    float max_val = -FLT_MAX;
    for (size_t j = 0; j < N; j++)
        max_val = fmaxf(max_val, row_in[j]);

    float sum = 0.0f;
    for (size_t j = 0; j < N; j++)
        sum += expf(row_in[j] - max_val);

    for (size_t j = 0; j < N; j++)
        row_out[j] = expf(row_in[j] - max_val) / sum;
}

extern "C" void solution(const float* A, float* C, size_t M, size_t N) {
    softmax_kernel<<<M, 1>>>(A, C, M, N);
    cudaDeviceSynchronize();
}
