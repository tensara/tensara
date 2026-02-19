#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

// Each block handles one "slice" along the softmax dimension.
// We flatten the tensor into (outer, dim_size, inner) and softmax over dim_size.
__global__ void softmax_kernel(const float* input, float* output,
                                size_t outer, size_t dim_size, size_t inner) {
    size_t outer_idx = blockIdx.x;
    size_t inner_idx = blockIdx.y;
    if (outer_idx >= outer || inner_idx >= inner) return;

    // Pointer to start of this slice
    const float* in_slice = input + outer_idx * dim_size * inner + inner_idx;
    float* out_slice = output + outer_idx * dim_size * inner + inner_idx;

    // Find max
    float max_val = -FLT_MAX;
    for (size_t i = 0; i < dim_size; i++)
        max_val = fmaxf(max_val, in_slice[i * inner]);

    // Compute sum of exp
    float sum = 0.0f;
    for (size_t i = 0; i < dim_size; i++)
        sum += expf(in_slice[i * inner] - max_val);

    // Write output
    for (size_t i = 0; i < dim_size; i++)
        out_slice[i * inner] = expf(in_slice[i * inner] - max_val) / sum;
}

extern "C" void solution(const float* input, int dim,
                          float* output,
                          const size_t* shape, size_t ndim) {
    // Compute outer, dim_size, inner
    size_t outer = 1, dim_size = shape[dim], inner = 1;
    for (int i = 0; i < dim; i++)        outer *= shape[i];
    for (size_t i = dim + 1; i < ndim; i++) inner *= shape[i];

    dim3 grid(outer, inner);
    softmax_kernel<<<grid, 1>>>(input, output, outer, dim_size, inner);
    cudaDeviceSynchronize();
}
