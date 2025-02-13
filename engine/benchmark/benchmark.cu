#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <memory>
#include <functional>
#include "solution.cuh"

template<typename T>
class BenchmarkRunner {
private:
    cudaEvent_t start, stop;
    std::vector<T*> d_inputs;
    std::vector<T*> d_outputs;
    size_t num_inputs;
    size_t num_outputs;
    size_t size;
    size_t num_runs;

public:
    BenchmarkRunner(
        size_t input_buffers,
        size_t output_buffers,
        size_t element_count,
        size_t runs = 100
    ) : num_inputs(input_buffers),
        num_outputs(output_buffers),
        size(element_count),
        num_runs(runs) {
        
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        d_inputs.resize(num_inputs);
        d_outputs.resize(num_outputs);
        for (size_t i = 0; i < num_inputs; i++) {
            void* ptr;
            cudaMalloc(&ptr, size * sizeof(T));
            d_inputs[i] = static_cast<T*>(ptr);
        }
        for (size_t i = 0; i < num_outputs; i++) {
            void* ptr;
            cudaMalloc(&ptr, size * sizeof(T));
            d_outputs[i] = static_cast<T*>(ptr);
        }
    }

    ~BenchmarkRunner() {
        for (size_t i = 0; i < d_inputs.size(); i++) {
            cudaFree(d_inputs[i]);
        }
        for (size_t i = 0; i < d_outputs.size(); i++) {
            cudaFree(d_outputs[i]);
        }
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void load_input_data(const std::vector<const T*>& input_data) {
        for (size_t buf = 0; buf < num_inputs; buf++) {
            cudaMemcpy(d_inputs[buf], input_data[buf], 
                      size * sizeof(T), cudaMemcpyHostToDevice);
        }
    }

    typedef void (*KernelLauncher)(const std::vector<T*>&, const std::vector<T*>&, size_t);
    
    void run_benchmark(KernelLauncher kernel_launcher) {
        float total_ms = 0.0f;
        float min_ms = 1e9;
        float max_ms = 0.0f;

        kernel_launcher(d_inputs, d_outputs, size);
        cudaDeviceSynchronize();

        for (size_t i = 0; i < num_runs; i++) {
            cudaEventRecord(start);
            kernel_launcher(d_inputs, d_outputs, size);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float ms = 0.0f;
            cudaEventElapsedTime(&ms, start, stop);
            total_ms += ms;
            min_ms = std::min(min_ms, ms);
            max_ms = std::max(max_ms, ms);
        }

        float avg_ms = total_ms / num_runs;
        float bytes_per_elem = sizeof(T) * (num_inputs + num_outputs);
        float gb_per_sec = (size * bytes_per_elem) / (avg_ms * 1e-3) / 1e9;

        std::cout << "\nBenchmark Results for size " << size << ":\n";
        std::cout << "----------------------------------------\n";
        std::cout << "Average Runtime: " << avg_ms << " ms\n";
        std::cout << "Min Runtime: " << min_ms << " ms\n";
        std::cout << "Max Runtime: " << max_ms << " ms\n";
        std::cout << "Memory Bandwidth: " << gb_per_sec << " GB/s\n";
        std::cout << "Throughput: " << (size / (avg_ms * 1e-3)) / 1e9 
                 << " billion elements/second\n";
    }
};

typedef void (*VectorAddLauncher)(const std::vector<float*>&, const std::vector<float*>&, size_t);

void launch_solution(const std::vector<float*>& inputs, 
                    const std::vector<float*>& outputs, 
                    size_t n) {
    solution(inputs[0], inputs[1], outputs[0], n);
}

int main() {
    size_t size = 1 << 24;
    
    std::vector<float*> host_inputs(2);
    for (size_t i = 0; i < host_inputs.size(); i++) {
        host_inputs[i] = new float[size];
    }
    
    for (size_t i = 0; i < size; i++) {
        host_inputs[0][i] = static_cast<float>(i);
        host_inputs[1][i] = static_cast<float>(i * 2);
    }
    
    std::vector<const float*> const_inputs;
    for (size_t i = 0; i < host_inputs.size(); i++) {    
        const_inputs.push_back(host_inputs[i]);
    }
    
    BenchmarkRunner<float> runner(2, 1, size);
    
    runner.load_input_data(const_inputs);
    runner.run_benchmark(launch_solution);
    
    for (size_t i = 0; i < host_inputs.size(); i++) {
        delete[] host_inputs[i];
    }
    
    return 0;
}
