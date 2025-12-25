#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TILE_DIM 16

// Simple error-checking macro
#define HIP_CHECK(command) { \
    hipError_t status = command; \
    if (status != hipSuccess) { \
        fprintf(stderr, "HIP Error: %s at %s:%d\n", \
                hipGetErrorString(status), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// Tiled Matrix Multiplication Kernel
__global__ void matmul_tiled(const float* A, const float* B, float* C, int N) {
    int row = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int col = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    __shared__ float sA[TILE_DIM][TILE_DIM];
    __shared__ float sB[TILE_DIM][TILE_DIM];

    float sum = 0.0f;

    for (int t = 0; t < N / TILE_DIM; ++t) {
        // Load tiles into shared memory
        int aRow = row;
        int aCol = t * TILE_DIM + hipThreadIdx_x;
        int bRow = t * TILE_DIM + hipThreadIdx_y;
        int bCol = col;

        if (aRow < N && aCol < N)
            sA[hipThreadIdx_y][hipThreadIdx_x] = A[aRow * N + aCol];
        else
            sA[hipThreadIdx_y][hipThreadIdx_x] = 0.0f;

        if (bRow < N && bCol < N)
            sB[hipThreadIdx_y][hipThreadIdx_x] = B[bRow * N + bCol];
        else
            sB[hipThreadIdx_y][hipThreadIdx_x] = 0.0f;

        __syncthreads();

        // Compute partial dot product
        for (int k = 0; k < TILE_DIM; ++k) {
            sum += sA[hipThreadIdx_y][k] * sB[k][hipThreadIdx_x];
        }
        
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// Host-side verification
void verify(const float* A, const float* B, const float* C, int N) {
    printf("Verifying result...\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float cpu_sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                cpu_sum += A[i * N + k] * B[k * N + j];
            }
            float gpu_val = C[i * N + j];
            float diff = fabsf(cpu_sum - gpu_val);
            if (diff > 1e-4) {
                printf("Verification FAILED! at (%d, %d)\n", i, j);
                printf("CPU: %f, GPU: %f, Diff: %f\n", cpu_sum, gpu_val, diff);
                return;
            }
        }
    }
    printf("Result: PASSED ✓\n");
}

// Main function
int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <N> (matrix size)\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    if (N == 0 || (N % TILE_DIM != 0)) {
        fprintf(stderr, "Matrix size N must be a multiple of %d\n", TILE_DIM);
        return 1;
    }

    size_t matrix_size = (size_t)N * N;
    size_t bytes = matrix_size * sizeof(float);

    printf("=== HIP Matrix Multiplication ===\n");
    printf("Matrix dimensions: A(%dx%d) * B(%dx%d) = C(%dx%d)\n", N, N, N, N, N, N);
    printf("Tile size: %dx%d\n", TILE_DIM, TILE_DIM);
    printf("Memory: %.2fMB per matrix\n\n", bytes / (1024.0 * 1024.0));

    // Allocate host memory
    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C = (float*)malloc(bytes);

    // Initialize host memory
    for (size_t i = 0; i < matrix_size; ++i) {
        h_A[i] = (float)rand() / (float)RAND_MAX;
        h_B[i] = (float)rand() / (float)RAND_MAX;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    HIP_CHECK(hipMalloc(&d_A, bytes));
    HIP_CHECK(hipMalloc(&d_B, bytes));
    HIP_CHECK(hipMalloc(&d_C, bytes));

    // Copy data from host to device
    HIP_CHECK(hipMemcpy(d_A, h_A, bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B, bytes, hipMemcpyHostToDevice));

    // Kernel launch parameters
    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 blocks(N / TILE_DIM, N / TILE_DIM);

    // Create HIP events for timing
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    // Record start event
    HIP_CHECK(hipEventRecord(start));

    // Launch kernel
    hipLaunchKernelGGL(matmul_tiled, blocks, threads, 0, 0, d_A, d_B, d_C, N);

    // Record stop event and synchronize
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    // Calculate kernel execution time
    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    
    double gflops = (2.0 * (double)N * (double)N * (double)N) / (ms / 1000.0) / 1e9;
    printf("Kernel execution: %.2f ms\n", ms);
    printf("Performance: %.2f GFLOPS\n\n", gflops);

    // Copy result from device to host
    HIP_CHECK(hipMemcpy(h_C, d_C, bytes, hipMemcpyDeviceToHost));

    // Verify the result
    verify(h_A, h_B, h_C, N);

    // Clean up
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
