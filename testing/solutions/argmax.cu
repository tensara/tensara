#include <cuda_runtime.h>
#include <float.h>

__global__ void argmax_kernel(const float* input, int* output,
                               int outer, int dim_size, int inner) {
    int outer_idx = blockIdx.x;
    int inner_idx = blockIdx.y;
    if (outer_idx >= outer || inner_idx >= inner) return;

    const float* slice = input + outer_idx * dim_size * inner + inner_idx;
    float max_val = -FLT_MAX;
    int max_idx = 0;
    for (int i = 0; i < dim_size; i++) {
        float val = slice[i * inner];
        if (val > max_val) { max_val = val; max_idx = i; }
    }
    output[outer_idx * inner + inner_idx] = max_idx;
}

extern "C" void solution(const float* input, int dim,
                          int* output, const int* shape, int ndim) {
    int dim_size = shape[dim];
    int outer = 1, inner = 1;
    for (int i = 0; i < dim; i++)       outer *= shape[i];
    for (int i = dim + 1; i < ndim; i++) inner *= shape[i];

    dim3 grid(outer, inner);
    argmax_kernel<<<grid, 1>>>(input, output, outer, dim_size, inner);
    cudaDeviceSynchronize();
}
