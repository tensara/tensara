#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "solution.cuh"

class BenchmarkRunner {
private:
    cudaEvent_t start, stop;
    float *d_input1, *d_input2, *d_output;
    float *h_input1, *h_input2, *h_output;  // Host arrays for verification
    size_t size;
    size_t num_runs;

public:
    BenchmarkRunner(size_t n, size_t runs = 100) 
        : size(n), num_runs(runs) {
        // Create CUDA events
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        // Allocate device memory
        cudaMalloc(&d_input1, size * sizeof(float));
        cudaMalloc(&d_input2, size * sizeof(float));
        cudaMalloc(&d_output, size * sizeof(float));
        
        // Allocate host memory
        h_input1 = new float[size];
        h_input2 = new float[size];
        h_output = new float[size];
        
        // Initialize input data
        for (size_t i = 0; i < size; i++) {
            h_input1[i] = static_cast<float>(i);
            h_input2[i] = static_cast<float>(i * 2);
        }
        
        // Copy data to device
        cudaMemcpy(d_input1, h_input1, size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_input2, h_input2, size * sizeof(float), cudaMemcpyHostToDevice);
    }

    ~BenchmarkRunner() {
        // Cleanup
        cudaFree(d_input1);
        cudaFree(d_input2);
        cudaFree(d_output);
        delete[] h_input1;
        delete[] h_input2;
        delete[] h_output;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    bool verify_result() {
        // Copy result back to host
        cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Verify first few and last few elements
        for (size_t i = 0; i < std::min(size_t(5), size); i++) {
            float expected = h_input1[i] + h_input2[i];
            if (abs(h_output[i] - expected) > 1e-5) {
                std::cout << "Verification failed at " << i << ": "
                          << h_output[i] << " != " << expected << "\n";
                return false;
            }
        }
        if (size > 5) {
            for (size_t i = size - 5; i < size; i++) {
                float expected = h_input1[i] + h_input2[i];
                if (abs(h_output[i] - expected) > 1e-5) {
                    std::cout << "Verification failed at " << i << ": "
                              << h_output[i] << " != " << expected << "\n";
                    return false;
                }
            }
        }
        return true;
    }

    void run_benchmark() {
        float total_ms = 0.0f;
        float min_ms = 1e9;
        float max_ms = 0.0f;

        // Warmup run
        solution(d_input1, d_input2, d_output, size);
        cudaDeviceSynchronize();

        // Benchmark runs
        for (size_t i = 0; i < num_runs; i++) {
            cudaEventRecord(start);
            solution(d_input1, d_input2, d_output, size);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float ms = 0.0f;
            cudaEventElapsedTime(&ms, start, stop);
            total_ms += ms;
            min_ms = std::min(min_ms, ms);
            max_ms = std::max(max_ms, ms);
        }

        // Verify result
        bool correct = verify_result();

        // Calculate and print results
        float avg_ms = total_ms / num_runs;
        float gb_per_sec = (size * 3 * sizeof(float)) / (avg_ms * 1e-3) / 1e9;  // 2 reads + 1 write
        
        std::cout << "\nBenchmark Results for size " << size << ":\n";
        std::cout << "----------------------------------------\n";
        std::cout << "Correctness: " << (correct ? "PASSED" : "FAILED") << "\n";
        std::cout << "Average Runtime: " << avg_ms << " ms\n";
        std::cout << "Min Runtime: " << min_ms << " ms\n";
        std::cout << "Max Runtime: " << max_ms << " ms\n";
        std::cout << "Memory Bandwidth: " << gb_per_sec << " GB/s\n";
        std::cout << "Throughput: " << (size / (avg_ms * 1e-3)) / 1e9 << " billion elements/second\n";
    }
};

int main(int argc, char** argv) {
    // Test different sizes
    std::vector<size_t> sizes = {1<<20, 1<<22, 1<<24, 1<<26};
    
    for (size_t size : sizes) {
        BenchmarkRunner runner(size);
        runner.run_benchmark();
    }
    
    return 0;
}
