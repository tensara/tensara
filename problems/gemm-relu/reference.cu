#include <cuda_runtime.h>

// Split into 3 extremely unoptimized kernels

__global__ void reference_gemm_kernel(float* A, float* W, float* temp,
                                    size_t batch_size, size_t in_features, size_t out_features) {
    // Process only one element per thread
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size * out_features) {
        size_t row = idx / out_features;
        size_t col = idx % out_features;
        
        volatile float sum = 0.0f;
        // Do computation in reverse order for worse memory access pattern
        for (size_t k = in_features; k > 0; k--) {
            float a = A[row * in_features + (k-1)];
            float w = W[col * in_features + (k-1)];
            sum += a * w;
        }
        temp[row * out_features + col] = sum;
    }
}

__global__ void reference_bias_kernel(float* temp, float* b, float* temp2,
                                    size_t batch_size, size_t out_features) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size * out_features) {
        size_t row = idx / out_features;
        size_t col = idx % out_features;
        
        temp2[row * out_features + col] = temp[row * out_features + col] + b[col];
    }
}

__global__ void reference_relu_kernel(float* temp2, float* C,
                                    size_t batch_size, size_t out_features) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size * out_features) {
        size_t row = idx / out_features;
        size_t col = idx % out_features;
        
        float val = temp2[row * out_features + col];
        C[row * out_features + col] = val > 0.0f ? val : 0.0f;
    }
}

extern "C" void reference_solution(float* A, float* W, float* b, float* C,
                                 size_t B, size_t N, size_t M) {
    // Allocate temporary buffers
    float *temp, *temp2;
    cudaMalloc(&temp, B * M * sizeof(float));
    cudaMalloc(&temp2, B * M * sizeof(float));
    
    // Use tiny block size for inefficiency
    dim3 block_size(4, 1);
    dim3 num_blocks((B * M + block_size.x - 1) / block_size.x, 1);

    // Call three separate kernels
    reference_gemm_kernel<<<num_blocks, block_size>>>(A, W, temp, B, N, M);
    cudaDeviceSynchronize();
    
    reference_bias_kernel<<<num_blocks, block_size>>>(temp, b, temp2, B, M);
    cudaDeviceSynchronize();
    
    reference_relu_kernel<<<num_blocks, block_size>>>(temp2, C, B, M);
    cudaDeviceSynchronize();
    
    // Clean up
    cudaFree(temp);
    cudaFree(temp2);
}