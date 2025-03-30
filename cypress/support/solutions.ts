export const ReluSolutions = {
  correct: "#include <cuda_runtime.h>\n\n__global__ void matrixReLU(float* input, float* output, size_t rows, size_t cols) {\n    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n    int row_index = idx / cols;\n    int col_index = idx % cols;\n\n    if (row_index < rows && col_index < cols) {\n        int linear_idx = row_index * cols + col_index;\n        output[linear_idx] = (input[linear_idx] > 0) ? input[linear_idx] : 0.0f;\n    }\n\n}\n\nextern \"C\" void solution(float* input, float* output, size_t n, size_t m) {\n    int threadsPerBlock = 256;\n    int blocksPerGrid = (n * m + threadsPerBlock - 1) / threadsPerBlock;\n    matrixReLU<<<blocksPerGrid, threadsPerBlock>>>(input, output, m, n);\n    return;\n    \n}",
  compile_error: "#include <cuda_runtime.h>\n\n__global__ void matrixReLU(float* input, float* output, size_t rows, size_t cols) {\n    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n    int row_index = idx / cols;\n    int col_index = idx % cols;\n\n    if (row_index < rows && col_index < cols) {\n        int linear_idx = row_index * cols + col_index;\n        output[linear_idx] = (input[linear_idx] > 0) ? input[linear_idx] : 0.0f;\n    }\n\n}\n\nextern \"C\" void solution(float* input, float* output, size_t n, size_t m) {\n    int threadsPerBlock = 256;\n    int blocksPerGrid = (n * m + threadsPerBlock - 1) / threadsPerBlock;\n    // Missing angle brackets will cause a compile error\n    matrixReLU(blocksPerGrid, threadsPerBlock)(input, output, m, n);\n    return;\n    \n}",
  runtime_error: "#include <cuda_runtime.h>\n\n__global__ void matrixReLU(float* input, float* output, size_t rows, size_t cols) {\n    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n    \n    // Will cause division by zero for certain thread indices\n    float test_value = 1.0f / (threadIdx.x % 32);\n    \n    int row_index = idx / cols;\n    int col_index = idx % cols;\n\n    if (row_index < rows && col_index < cols) {\n        int linear_idx = row_index * cols + col_index;\n        // Using potentially infinite value in calculation\n        output[linear_idx] = (input[linear_idx] > 0) ? input[linear_idx] * test_value : 0.0f;\n    }\n\n}\n\nextern \"C\" void solution(float* input, float* output, size_t n, size_t m) {\n     int threadsPerBlock = 256;\n    int blocksPerGrid = (n * m + threadsPerBlock - 1) / threadsPerBlock;\n    matrixReLU<<<blocksPerGrid, threadsPerBlock>>>(input, output, m, n);\n    return;\n    \n}",
  wrong_answer: "#include <cuda_runtime.h>\n\n__global__ void matrixReLU(float* input, float* output, size_t rows, size_t cols) {\n    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n    int row_index = idx / cols;\n    int col_index = idx % cols;\n\n    if (row_index < rows && col_index < cols) {\n        int linear_idx = row_index * cols + col_index;\n        // Incorrect ReLU implementation (negating positive values)\n        output[linear_idx] = (input[linear_idx] > 0) ? 0.0f : input[linear_idx];\n    }\n\n}\n\nextern \"C\" void solution(float* input, float* output, size_t n, size_t m) {\n    int threadsPerBlock = 256;\n    int blocksPerGrid = (n * m + threadsPerBlock - 1) / threadsPerBlock;\n    // Parameters in wrong order\n    matrixReLU<<<blocksPerGrid, threadsPerBlock>>>(input, output, n, m);\n    return;\n    \n}"
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