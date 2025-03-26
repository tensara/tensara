#include <cuda_runtime.h>

size_t cdiv(size_t x, size_t y) {
    return (x+y-1)/y;
}

__global__ void kernel(const float*__restrict__ A, const float*__restrict__ B, float* C, size_t H, size_t W, size_t Kh, size_t Kw) {
    extern __shared__ float sB[];
    for(size_t i = threadIdx.y * 32 + threadIdx.x; i < Kh * Kw; i += blockDim.y * blockDim.x) {
        sB[i] = B[i];
    }
    __syncthreads();
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < H && j < W) {
        float acc = 0;
        for(size_t k = 0; k < Kh; k++) {
            if(i + k < (Kh - 1) / 2 || i + k - (Kh - 1) / 2 >= H) {
                continue;
            }
            size_t ak = i + k - (Kh - 1) / 2;
            for(size_t l = 0; l < Kw; l++) {
                if(j + l < (Kw - 1) / 2 || j + l - (Kw - 1) / 2 >= W) {
                    continue;
                }
                size_t al = j + l - (Kw - 1) / 2;
                acc += A[ak * W + al] * sB[k * Kw + l];
            }
        }
        C[i * W + j] = acc;
    }
}

// Note: A, B, and C are all device pointers to float arrays
extern "C" void solution(float* A, float* B, float* C, size_t H, size_t W, size_t Kh, size_t Kw) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Kh*Kw*4);
    kernel<<<dim3(cdiv(H,32),cdiv(W,32)),dim3(32,32),Kh*Kw*4>>>(A, B, C, H, W, Kh, Kw);
}