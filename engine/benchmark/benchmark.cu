#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <memory>
#include <functional>
#include <iomanip>
#include "test_cases.hpp"
#include "problem_test.hpp"

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
    BenchmarkRunner(const TestCase<T>& test_case, size_t runs = 10) 
        : num_inputs(test_case.get_num_inputs()),
          num_outputs(test_case.get_num_outputs()),
          size(test_case.get_size()),
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

        // Load the test case data immediately
        load_input_data(test_case.get_inputs());
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

private:
    void load_input_data(const std::vector<const T*>& input_data) {
        for (size_t buf = 0; buf < num_inputs; buf++) {
            cudaMemcpy(d_inputs[buf], input_data[buf], 
                      size * sizeof(T), cudaMemcpyHostToDevice);
        }
    }

public:
    typedef void (*KernelLauncher)(const std::vector<T*>&, const std::vector<T*>&, size_t);
    
    float run_benchmark(KernelLauncher kernel_launcher) {
        float total_ms = 0.0f;

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
        }

        return total_ms / num_runs;
    }
};

int main() {
    auto test_cases = create_test_cases();
    
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Running benchmarks..." << std::endl;
    std::cout << "Size\t\tTime (ms)" << std::endl;
    std::cout << "------------------------" << std::endl;
    
    for (auto test_case : test_cases) {
        BenchmarkRunner<float> runner(*test_case);
        float avg_ms = runner.run_benchmark(launch_kernel);
        std::cout << test_case->get_size() << "\t\t" << avg_ms << std::endl;
        delete test_case;
    }
    
    return 0;
}
