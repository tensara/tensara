#include <hip/hip_runtime.h>
#include <stdio.h>

// Note: input, output are all device pointers to float32 arrays
extern "C" __global__ void solution(const float* input, float alpha, float* output, size_t n, size_t m) {
    // Calculate global thread index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_elements = n * m;
    
    // Boundary check
    if (idx < total_elements) {
        float val = input[idx];
        // Leaky ReLU: max(alpha * val, val)
        // Equivalent to: val > 0 ? val : alpha * val
        output[idx] = val > 0.0f ? val : alpha * val;
    }
}

// Simple test harness
int main() {
    printf("=== HIP Leaky ReLU Test ===\n");
    
    size_t n = 1024;
    size_t m = 1024;
    size_t total = n * m;
    size_t bytes = total * sizeof(float);
    float alpha = 0.01f;
    
    printf("Matrix size: %zu x %zu (%zu elements)\n", n, m, total);
    printf("Alpha: %.2f\n", alpha);
    printf("Memory: %.2f MB\n\n", bytes / (1024.0 * 1024.0));
    
    // Allocate host memory
    float *h_input = (float*)malloc(bytes);
    float *h_output = (float*)malloc(bytes);
    
    // Initialize test data
    for (size_t i = 0; i < total; i++) {
        h_input[i] = (float)i - (float)(total / 2);  // Mix of positive and negative
    }
    
    // Allocate device memory
    float *d_input, *d_output;
    hipMalloc(&d_input, bytes);
    hipMalloc(&d_output, bytes);
    
    // Copy to device
    hipMemcpy(d_input, h_input, bytes, hipMemcpyHostToDevice);
    
    // Launch kernel
    dim3 threads(256);
    dim3 blocks((total + threads.x - 1) / threads.x);
    
    printf("Launching kernel with %d blocks x %d threads\n", blocks.x, threads.x);
    
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    
    hipEventRecord(start);
    hipLaunchKernelGGL(solution, blocks, threads, 0, 0, d_input, alpha, d_output, n, m);
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    
    float ms = 0.0f;
    hipEventElapsedTime(&ms, start, stop);
    
    printf("Kernel execution: %.3f ms\n", ms);
    
    // Copy result back
    hipMemcpy(h_output, d_output, bytes, hipMemcpyDeviceToHost);
    
    // Verify results
    printf("\nVerifying results...\n");
    int errors = 0;
    for (size_t i = 0; i < total && errors < 10; i++) {
        float expected = h_input[i] > 0.0f ? h_input[i] : alpha * h_input[i];
        float diff = fabsf(h_output[i] - expected);
        if (diff > 1e-5) {
            printf("ERROR at index %zu: expected %.6f, got %.6f (diff %.6f)\n", 
                   i, expected, h_output[i], diff);
            errors++;
        }
    }
    
    if (errors == 0) {
        printf("Result: PASSED ✓\n");
        printf("\nAll %zu elements verified successfully!\n", total);
    } else {
        printf("Result: FAILED ✗\n");
        printf("Found %d errors\n", errors);
    }
    
    // Cleanup
    hipFree(d_input);
    hipFree(d_output);
    free(h_input);
    free(h_output);
    
    return errors > 0 ? 1 : 0;
}
