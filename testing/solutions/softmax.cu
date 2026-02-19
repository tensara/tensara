#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

// One block per row
__global__ void softmax_kernel(const float* A, float* C, int M, int N) {
    int row = blockIdx.x;
    if (row >= M) return;

    const float* row_in  = A + row * N;
    float*       row_out = C + row * N;

    // Find max for numerical stability
    float max_val = -FLT_MAX;
    for (int j = 0; j < N; j++)
        max_val = fmaxf(max_val, row_in[j]);

    // Sum of exp
    float sum = 0.0f;
    for (int j = 0; j < N; j++)
        sum += expf(row_in[j] - max_val);

    for (int j = 0; j < N; j++)
        row_out[j] = expf(row_in[j] - max_val) / sum;
}

extern "C" void solution(const float* A, float* C, int M, int N) {
    softmax_kernel<<<M, 1>>>(A, C, M, N);
    cudaDeviceSynchronize();
}
