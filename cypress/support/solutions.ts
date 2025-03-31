export const ReluSolutions = {
  correct: `#include <cuda_runtime.h>

__global__ void matrixReLU(float* input, float* output, size_t rows, size_t cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_index = idx / cols;
    int col_index = idx % cols;

    if (row_index < rows && col_index < cols) {
        int linear_idx = row_index * cols + col_index;
        output[linear_idx] = (input[linear_idx] > 0) ? input[linear_idx] : 0.0f;
    }
}

extern "C" void solution(float* input, float* output, size_t n, size_t m) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n * m + threadsPerBlock - 1) / threadsPerBlock;
    matrixReLU<<<blocksPerGrid, threadsPerBlock>>>(input, output, m, n);
    return;
}`,

  compile_error: `#include <cuda_runtime.h>

__global__ void matrixReLU(float* input, float* output, size_t rows, size_t cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_index = idx / cols;
    int col_index = idx % cols;

    if (row_index < rows && col_index < cols) {
        int linear_idx = row_index * cols + col_index;
        output[linear_idx] = (input[linear_idx] > 0) ? input[linear_idx] : 0.0f;
    }
}

extern "C" void solution(float* input, float* output, size_t n, size_t m) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n * m + threadsPerBlock - 1) / threadsPerBlock;
    // Missing angle brackets will cause a compile error
    matrixReLU(blocksPerGrid, threadsPerBlock)(input, output, m, n);
    return;
}`,

  runtime_error: `#include <cuda_runtime.h>

__global__ void matrixReLU(float* input, float* output, size_t rows, size_t cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Will cause division by zero for certain thread indices
    float test_value = 1.0f / (threadIdx.x % 32);
    
    int row_index = idx / cols;
    int col_index = idx % cols;

    if (row_index < rows && col_index < cols) {
        int linear_idx = row_index * cols + col_index;
        // Using potentially infinite value in calculation
        output[linear_idx] = (input[linear_idx] > 0) ? input[linear_idx] * test_value : 0.0f;
    }
}

extern "C" void solution(float* input, float* output, size_t n, size_t m) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n * m + threadsPerBlock - 1) / threadsPerBlock;
    matrixReLU<<<blocksPerGrid, threadsPerBlock>>>(input, output, m, n);
    return;
}`,

  wrong_answer: `#include <cuda_runtime.h>

__global__ void matrixReLU(float* input, float* output, size_t rows, size_t cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_index = idx / cols;
    int col_index = idx % cols;

    if (row_index < rows && col_index < cols) {
        int linear_idx = row_index * cols + col_index;
        // Incorrect ReLU implementation (negating positive values)
        output[linear_idx] = (input[linear_idx] > 0) ? 0.0f : input[linear_idx];
    }
}

extern "C" void solution(float* input, float* output, size_t n, size_t m) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n * m + threadsPerBlock - 1) / threadsPerBlock;
    // Parameters in wrong order
    matrixReLU<<<blocksPerGrid, threadsPerBlock>>>(input, output, n, m);
    return;
}`
}

export const Conv1DSolutions = {
  correct: `#include <cuda_runtime.h>

__global__ void reference_conv1d_kernel(float* A, float* B, float* C, size_t N, size_t K) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float sum = 0.0f;
        int offset = (K - 1) / 2;
        for (size_t j = 0; j < K; j++) {
            int input_idx = idx + j - offset;
            float input_val = 0.0f;
            if (input_idx >= 0 && input_idx < N) {
                input_val = A[input_idx];
            }
            sum += input_val * B[j];
        }
        C[idx] = sum;
    }
}

extern "C" void solution(float* A, float* B, float* C, size_t N, size_t K) {
    int block_size = 256;
    int num_blocks = (N + block_size - 1) / block_size;
    reference_conv1d_kernel<<<num_blocks, block_size>>>(A, B, C, N, K);
    cudaDeviceSynchronize();
}`,

  compile_error: `#include <cuda_runtime.h>

__global__ void reference_conv1d_kernel(float* A, float* B, float* C, size_t N, size_t K) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float sum = 0.0f;
        int offset = (K - 1) / 2;
        for (size_t j = 0; j < K; j++) {
            int input_idx = idx + j - offset;
            float input_val = 0.0f;
            if (input_idx >= 0 && input_idx < N) {
                input_val = A[input_idx];
            }
            sum += input_val * B[j];
        }
        C[idx] = sum;
    }
}

extern "C" void solution(float* A, float* B, float* C, size_t N, size_t K) {
    int block_size = 256;
    int num_blocks = (N + block_size - 1) / block_size;
    // Missing angle brackets will cause a compile error
    reference_conv1d_kernel(num_blocks, block_size)(A, B, C, N, K);
    cudaDeviceSynchronize();
}`,

  runtime_error: `#include <cuda_runtime.h>

__global__ void reference_conv1d_kernel(float* A, float* B, float* C, size_t N, size_t K) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Will cause division by zero for certain thread indices
    float test_value = 1.0f / (threadIdx.x % 32);
    
    if (idx < N) {
        float sum = 0.0f;
        int offset = (K - 1) / 2;
        for (size_t j = 0; j < K; j++) {
            int input_idx = idx + j - offset;
            float input_val = 0.0f;
            if (input_idx >= 0 && input_idx < N) {
                input_val = A[input_idx];
            }
            // Using potentially infinite value in calculation
            sum += input_val * B[j] * test_value;
        }
        C[idx] = sum;
    }
}

extern "C" void solution(float* A, float* B, float* C, size_t N, size_t K) {
    int block_size = 256;
    int num_blocks = (N + block_size - 1) / block_size;
    reference_conv1d_kernel<<<num_blocks, block_size>>>(A, B, C, N, K);
    cudaDeviceSynchronize();
}`,

  wrong_answer: `#include <cuda_runtime.h>

__global__ void reference_conv1d_kernel(float* A, float* B, float* C, size_t N, size_t K) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float sum = 0.0f;
        // Incorrect offset calculation - wrong centering of kernel
        int offset = K / 2;  // Should be (K - 1) / 2
        for (size_t j = 0; j < K; j++) {
            int input_idx = idx + j - offset;
            float input_val = 0.0f;
            if (input_idx >= 0 && input_idx < N) {
                input_val = A[input_idx];
            }
            // Incorrect convolution - using subtraction instead of multiplication
            sum += input_val - B[j];  // Should be input_val * B[j]
        }
        C[idx] = sum;
    }
}

extern "C" void solution(float* A, float* B, float* C, size_t N, size_t K) {
    int block_size = 256;
    int num_blocks = (N + block_size - 1) / block_size;
    reference_conv1d_kernel<<<num_blocks, block_size>>>(A, B, C, N, K);
    cudaDeviceSynchronize();
}`
}

export const HuberLossSolutions = {
  correct: `#include <cuda_runtime.h>

__global__ void huberLossKernel(const float* predictions, const float* targets, float* output, size_t n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (idx < n) {
        float diff = predictions[idx] - targets[idx];
        float abs_diff = fabsf(diff);
        
        if (abs_diff < 1.0f) {
            output[idx] = 0.5f * diff * diff;
        } else {
            output[idx] = abs_diff - 0.5f;
        }
    }
}

extern "C" void solution(const float* predictions, const float* targets, float* output, size_t n) {
    const int threadsPerBlock = 256;
    const int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    huberLossKernel<<<numBlocks, threadsPerBlock>>>(predictions, targets, output, n);
}`,

  compile_error: `#include <cuda_runtime.h>

__global__ void huberLossKernel(const float* predictions, const float* targets, float* output, size_t n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (idx < n) {
        float diff = predictions[idx] - targets[idx];
        float abs_diff = fabsf(diff);
        
        if (abs_diff < 1.0f) {
            output[idx] = 0.5f * diff * diff;
        } else {
            output[idx] = abs_diff - 0.5f;
        }
    }
}

extern "C" void solution(const float* predictions, const float* targets, float* output, size_t n) {
    const int threadsPerBlock = 256;
    const int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    // Missing angle brackets will cause a compile error
    huberLossKernel(numBlocks, threadsPerBlock)(predictions, targets, output, n);
}`,

  runtime_error: `#include <cuda_runtime.h>

__global__ void huberLossKernel(const float* predictions, const float* targets, float* output, size_t n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    // Will cause division by zero for certain thread indices
    float divisor = threadIdx.x % 32;
    float factor = 1.0f / divisor;
    
    if (idx < n) {
        float diff = predictions[idx] - targets[idx];
        float abs_diff = fabsf(diff);
        
        if (abs_diff < 1.0f) {
            // Using potentially infinite value in calculation
            output[idx] = 0.5f * diff * diff * factor;
        } else {
            output[idx] = abs_diff - 0.5f;
        }
    }
}

extern "C" void solution(const float* predictions, const float* targets, float* output, size_t n) {
    const int threadsPerBlock = 256;
    const int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    huberLossKernel<<<numBlocks, threadsPerBlock>>>(predictions, targets, output, n);
}`,

  wrong_answer: `#include <cuda_runtime.h>

__global__ void huberLossKernel(const float* predictions, const float* targets, float* output, size_t n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (idx < n) {
        float diff = predictions[idx] - targets[idx];
        float abs_diff = fabsf(diff);
        
        // Wrong threshold value (using 0.5 instead of 1.0)
        if (abs_diff < 0.5f) {
            output[idx] = 0.5f * diff * diff;
        } else {
            // Wrong formula for the second case (missing 0.5 subtraction)
            output[idx] = abs_diff;
        }
    }
}

extern "C" void solution(const float* predictions, const float* targets, float* output, size_t n) {
    const int threadsPerBlock = 256;
    const int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    huberLossKernel<<<numBlocks, threadsPerBlock>>>(predictions, targets, output, n);
}`
}