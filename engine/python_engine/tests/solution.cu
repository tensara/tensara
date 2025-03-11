#include <cuda_runtime.h>

// Tile size for shared memory optimization
#define TILE_SIZE 32

__global__ void optimized_matrix_multiply(float* A, float* B, float* C, size_t N) {
    // Shared memory tiles
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];
    
    // Calculate global row and column
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    // Local thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Accumulator for dot product
    float sum = 0.0f;
    
    // Loop over tiles
    for (int tile = 0; tile < (N + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Collaborative loading of A and B tiles into shared memory
        if (row < N && (tile * TILE_SIZE + tx) < N)
            s_A[ty][tx] = A[row * N + tile * TILE_SIZE + tx];
        else
            s_A[ty][tx] = 0.0f;
        
        if ((tile * TILE_SIZE + ty) < N && col < N)
            s_B[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col];
        else
            s_B[ty][tx] = 0.0f;
        
        // Synchronize to ensure tiles are loaded
        __syncthreads();
        
        // Compute partial dot product for this tile
        #pragma unroll 8
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += s_A[ty][k] * s_B[k][tx];
        }
        
        // Synchronize before loading next tile
        __syncthreads();
    }
    
    // Write result to global memory
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// If matrix size is known at compile time and is a multiple of 32,
// this specialized version will be even faster
template<size_t N>
__global__ void optimized_matrix_multiply_fixed(float* A, float* B, float* C) {
    // Shared memory tiles
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];
    
    // Calculate global row and column
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    // Local thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Accumulator for dot product
    float sum = 0.0f;
    
    // Loop over tiles
    #pragma unroll 4
    for (int tile = 0; tile < N / TILE_SIZE; ++tile) {
        // Collaborative loading of A and B tiles into shared memory
        s_A[ty][tx] = A[row * N + tile * TILE_SIZE + tx];
        s_B[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col];
        
        // Synchronize to ensure tiles are loaded
        __syncthreads();
        
        // Compute partial dot product for this tile
        #pragma unroll 8
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += s_A[ty][k] * s_B[k][tx];
        }
        
        // Synchronize before loading next tile
        __syncthreads();
    }
    
    // Write result to global memory
    C[row * N + col] = sum;
}

// Larger matrices benefit from using float4 for vectorized memory access
__global__ void optimized_matrix_multiply_vectorized(float* A, float* B, float* C, size_t N) {
    // Shared memory tiles with padding to avoid bank conflicts
    __shared__ float s_A[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE + 1];
    
    // Calculate global row and column
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    // Local thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Accumulator for dot product
    float sum = 0.0f;
    
    // Loop over tiles
    for (int tile = 0; tile < (N + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Collaborative loading of A and B tiles into shared memory with vectorized loads where possible
        if (row < N && (tile * TILE_SIZE + tx) < N) {
            if ((tile * TILE_SIZE + tx + 3) < N && (tx % 4 == 0)) {
                // Vector load when aligned
                float4 tmp = *((float4*)(&A[row * N + tile * TILE_SIZE + tx]));
                s_A[ty][tx] = tmp.x;
                s_A[ty][tx+1] = tmp.y;
                s_A[ty][tx+2] = tmp.z;
                s_A[ty][tx+3] = tmp.w;
            } else {
                // Regular load otherwise
                s_A[ty][tx] = A[row * N + tile * TILE_SIZE + tx];
            }
        } else {
            s_A[ty][tx] = 0.0f;
        }
        
        if ((tile * TILE_SIZE + ty) < N && col < N) {
            if ((col % 4 == 0) && (col + 3) < N) {
                // Vector load when aligned
                float4 tmp = *((float4*)(&B[(tile * TILE_SIZE + ty) * N + col]));
                s_B[ty][tx] = tmp.x;
                if (tx + 1 < TILE_SIZE) s_B[ty][tx+1] = tmp.y;
                if (tx + 2 < TILE_SIZE) s_B[ty][tx+2] = tmp.z;
                if (tx + 3 < TILE_SIZE) s_B[ty][tx+3] = tmp.w;
            } else {
                // Regular load otherwise
                s_B[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col];
            }
        } else {
            s_B[ty][tx] = 0.0f;
        }
        
        // Synchronize to ensure tiles are loaded
        __syncthreads();
        
        // Compute partial dot product for this tile
        #pragma unroll 8
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += s_A[ty][k] * s_B[k][tx];
        }
        
        // Synchronize before loading next tile
        __syncthreads();
    }
    
    // Write result to global memory
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

extern "C" void solution(float* input_a, float* input_b, float* output_c, size_t n) {
    // Determine the best kernel and parameters based on matrix size
    if (n <= 1024) {
        // For small matrices, use the fixed size kernel if size matches
        if (n == 1024) {
            dim3 blockDim(TILE_SIZE, TILE_SIZE);
            dim3 gridDim(n / TILE_SIZE, n / TILE_SIZE);
            optimized_matrix_multiply_fixed<1024><<<gridDim, blockDim>>>(input_a, input_b, output_c);
        } 
        else if (n == 512) {
            dim3 blockDim(TILE_SIZE, TILE_SIZE);
            dim3 gridDim(n / TILE_SIZE, n / TILE_SIZE);
            optimized_matrix_multiply_fixed<512><<<gridDim, blockDim>>>(input_a, input_b, output_c);
        }
        else {
            // Use the basic optimized kernel for other small sizes
            dim3 blockDim(TILE_SIZE, TILE_SIZE);
            dim3 gridDim((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);
            optimized_matrix_multiply<<<gridDim, blockDim>>>(input_a, input_b, output_c, n);
        }
    } 
    else {
        // For large matrices, use the vectorized kernel
        dim3 blockDim(TILE_SIZE, TILE_SIZE);
        dim3 gridDim((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);
        optimized_matrix_multiply_vectorized<<<gridDim, blockDim>>>(input_a, input_b, output_c, n);
    }
}