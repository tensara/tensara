#include <cuda_runtime.h>
#include <cmath>

/**
 * CUDA Implementation of Swish Activation Function
 * =================================================
 * 
 * Problem: Compute swish(x) = x * sigmoid(x) for all elements in a matrix
 * where sigmoid(x) = 1 / (1 + exp(-x))
 * 
 * This implementation demonstrates several GPU optimization techniques:
 * 1. Coalesced memory access patterns
 * 2. Efficient thread block sizing
 * 3. Fast math operations (expf)
 * 4. Numerical stability considerations
 * 5. Proper boundary handling
 */

// Thread block size - chosen for optimal occupancy on most modern GPUs
// 256 threads provides good balance between occupancy and resource usage
#define BLOCK_SIZE 256

/**
 * Device function: Numerically stable sigmoid implementation
 * 
 * The sigmoid function can overflow for large negative values.
 * This implementation handles numerical stability by splitting into cases:
 * - For x >= 0: Use standard formula 1 / (1 + exp(-x))
 * - For x < 0: Use equivalent form exp(x) / (1 + exp(x))
 * 
 * This prevents overflow while maintaining accuracy across the full input range.
 * 
 * @param x Input value
 * @return sigmoid(x) in range (0, 1)
 */
__device__ __forceinline__ float sigmoid(float x) {
    // For numerical stability, we handle positive and negative cases differently
    if (x >= 0.0f) {
        // Standard case: sigmoid(x) = 1 / (1 + exp(-x))
        // Works well for positive x as exp(-x) won't overflow
        return 1.0f / (1.0f + expf(-x));
    } else {
        // For negative x: sigmoid(x) = exp(x) / (1 + exp(x))
        // This form prevents overflow since exp(x) is small when x is negative
        float exp_x = expf(x);
        return exp_x / (1.0f + exp_x);
    }
}

/**
 * CUDA Kernel: Swish activation function
 * 
 * Applies swish(x) = x * sigmoid(x) element-wise to the input array.
 * 
 * Memory Access Pattern:
 * - Uses 1D grid-stride loop for flexibility with large arrays
 * - Consecutive threads access consecutive memory locations (coalesced access)
 * - Each thread processes elements at stride 'gridSize' to handle arrays larger than grid
 * 
 * Thread Organization:
 * - 1D thread blocks of size BLOCK_SIZE (256 threads)
 * - Grid size calculated to cover all elements with good occupancy
 * 
 * @param input  Device pointer to input array (read-only)
 * @param output Device pointer to output array (write-only)
 * @param total_elements Total number of elements to process (n * m)
 */
__global__ void swish_kernel(const float* input, float* output, size_t total_elements) {
    // Calculate global thread index
    // blockIdx.x * blockDim.x gives the starting index for this block
    // threadIdx.x adds the thread's position within the block
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate grid stride (total number of threads in the grid)
    // This allows us to process arrays larger than the number of threads
    size_t stride = gridDim.x * blockDim.x;
    
    // Grid-stride loop: Each thread processes multiple elements if needed
    // This pattern provides:
    // 1. Flexibility - works for any array size
    // 2. Efficiency - maintains coalesced memory access
    // 3. Scalability - adapts to available GPU resources
    for (size_t i = idx; i < total_elements; i += stride) {
        // Read input value (coalesced memory access)
        float x = input[i];
        
        // Compute swish: x * sigmoid(x)
        // The sigmoid function handles numerical stability internally
        float sig = sigmoid(x);
        
        // Write result (coalesced memory access)
        output[i] = x * sig;
    }
}

/**
 * Host function: Entry point for swish activation
 * 
 * This function:
 * 1. Calculates optimal grid dimensions
 * 2. Launches the CUDA kernel
 * 3. Handles synchronization (implicit via kernel launch)
 * 
 * Performance Considerations:
 * - Uses 256 threads per block (optimal for most modern GPUs)
 * - Grid size ensures good occupancy without over-provisioning
 * - Kernel launch overhead is amortized over many elements
 * 
 * @param input  Device pointer to input matrix (row-major, size n*m)
 * @param output Device pointer to output matrix (row-major, size n*m)
 * @param n      Number of rows in the matrix
 * @param m      Number of columns in the matrix
 */
extern "C" void solution(const float* input, float* output, size_t n, size_t m) {
    // Calculate total number of elements
    size_t total_elements = n * m;
    
    // Handle edge case: empty matrix
    if (total_elements == 0) {
        return;
    }
    
    // Calculate grid dimensions
    // We want enough blocks to cover all elements, but not too many to avoid overhead
    // Formula: ceil(total_elements / BLOCK_SIZE)
    // Using integer arithmetic: (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE
    size_t num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Launch kernel with calculated dimensions
    // <<<num_blocks, BLOCK_SIZE>>> specifies:
    // - num_blocks: number of thread blocks in the grid
    // - BLOCK_SIZE: number of threads per block (256)
    // 
    // This creates a 1D grid of 1D blocks, which is optimal for:
    // - Simple element-wise operations
    // - Coalesced memory access patterns
    // - Minimal indexing overhead
    swish_kernel<<<num_blocks, BLOCK_SIZE>>>(input, output, total_elements);
    
    // Note: Kernel launch is asynchronous
    // If synchronization is needed, call cudaDeviceSynchronize()
    // For this interface, we assume the caller handles synchronization
}

/**
 * Performance Notes:
 * ===================
 * 
 * 1. Memory Bandwidth:
 *    - Each element requires 1 read + 1 write = 8 bytes of memory traffic
 *    - For large matrices, performance is likely memory-bound
 *    - Coalesced access patterns maximize memory bandwidth utilization
 * 
 * 2. Compute Efficiency:
 *    - The expf() function is optimized on NVIDIA GPUs
 *    - Fast math operations can be enabled with -use_fast_math compiler flag
 *    - The sigmoid computation is compute-intensive but well-optimized
 * 
 * 3. Occupancy:
 *    - 256 threads per block provides high occupancy on most GPUs
 *    - Low register usage allows multiple blocks per SM
 *    - No shared memory usage simplifies occupancy calculations
 * 
 * 4. Scalability:
 *    - Grid-stride loop pattern handles arbitrary matrix sizes
 *    - Works efficiently for both small and large matrices
 *    - Minimal overhead for kernel launch
 * 
 * Potential Optimizations:
 * ========================
 * 
 * For very large matrices:
 * - Consider using streams for overlapping computation and data transfer
 * - Implement tiling strategies for better cache utilization
 * 
 * For specific GPU architectures:
 * - Tune BLOCK_SIZE based on SM architecture (128, 256, 512)
 * - Use __launch_bounds__ to hint register usage
 * 
 * For mixed precision:
 * - Consider half-precision (FP16) for increased throughput on Tensor Cores
 * - Use __half or __half2 types for 2x memory bandwidth
 */