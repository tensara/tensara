#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

__global__ void softmax_kernel(const float* input, float* output,
                                size_t outer, size_t dim_size, size_t inner) {
    size_t outer_idx = blockIdx.x;
    size_t inner_idx = blockIdx.y;
    if (outer_idx >= outer || inner_idx >= inner) return;

    const float* in_slice  = input  + outer_idx * dim_size * inner + inner_idx;
    float*       out_slice = output + outer_idx * dim_size * inner + inner_idx;

    float max_val = -FLT_MAX;
    for (size_t i = 0; i < dim_size; i++)
        max_val = fmaxf(max_val, in_slice[i * inner]);

    float sum = 0.0f;
    for (size_t i = 0; i < dim_size; i++)
        sum += expf(in_slice[i * inner] - max_val);

    for (size_t i = 0; i < dim_size; i++)
        out_slice[i * inner] = expf(in_slice[i * inner] - max_val) / sum;
}

extern "C" void solution(const float* input, int dim,
                          float* output,
                          const size_t* shape, size_t ndim) {
    // shape is a host pointer — compute outer/dim_size/inner on CPU
    size_t dim_size = shape[dim];
    size_t outer = 1, inner = 1;
    for (int i = 0; i < dim; i++)          outer *= shape[i];
    for (size_t i = dim + 1; i < ndim; i++) inner *= shape[i];

    dim3 grid((unsigned)outer, (unsigned)inner);
    softmax_kernel<<<grid, 1>>>(input, output, outer, dim_size, inner);
    cudaDeviceSynchronize();
}
