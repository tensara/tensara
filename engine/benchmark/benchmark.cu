#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <memory>
#include <functional>
#include <iomanip>
#include <utility>
#include "core.hpp"
#include "tests.hpp"

template<typename T>
class BenchmarkRunner {
private:
    cudaEvent_t start, stop;
    std::vector<T*> h_inputs;
    std::vector<T*> h_outputs;
    std::vector<T*> d_inputs;
    std::vector<T*> d_outputs;
    size_t num_runs;

public:
    BenchmarkRunner(TestCase<T>& test_case, size_t runs = 10) 
        : num_runs(runs) {
        
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        h_inputs.resize(test_case.input_shapes().size());
        h_outputs.resize(test_case.output_shapes().size());
        
        for (size_t i = 0; i < test_case.input_shapes().size(); i++) {
            size_t size = test_case.input_shapes()[i]->size();
            h_inputs[i] = new T[size];
        }
        
        for (size_t i = 0; i < test_case.output_shapes().size(); i++) {
            size_t size = test_case.output_shapes()[i]->size();
            h_outputs[i] = new T[size];
        }

        test_case.prepare_data(h_inputs.data(), h_outputs.data());

        d_inputs.resize(test_case.input_shapes().size());
        d_outputs.resize(test_case.output_shapes().size());
        
        for (size_t i = 0; i < test_case.input_shapes().size(); i++) {
            size_t size = test_case.input_shapes()[i]->size();
            cudaMalloc(&d_inputs[i], size * sizeof(T));
            cudaMemcpy(d_inputs[i], h_inputs[i], size * sizeof(T), cudaMemcpyHostToDevice);
        }
        
        for (size_t i = 0; i < test_case.output_shapes().size(); i++) {
            size_t size = test_case.output_shapes()[i]->size();
            cudaMalloc(&d_outputs[i], size * sizeof(T));
        }
    }

    ~BenchmarkRunner() {
        for (auto ptr : h_inputs) delete[] ptr;
        for (auto ptr : h_outputs) delete[] ptr;
        
        for (auto ptr : d_inputs) cudaFree(ptr);
        for (auto ptr : d_outputs) cudaFree(ptr);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

public:
    typedef void (*KernelLauncher)(const std::vector<T*>&, const std::vector<T*>&, const std::vector<size_t>&);
    
    std::pair<size_t, float> run_benchmark(KernelLauncher kernel_launcher, const TestCase<T>& test_case) {
        float total_ms = 0.0f;
        size_t flops = test_case.calculate_flops();
        std::vector<size_t> sizes = test_case.get_sizes();

        kernel_launcher(d_inputs, d_outputs, sizes);
        cudaDeviceSynchronize();

        for (size_t i = 0; i < num_runs; i++) {
            cudaEventRecord(start);
            kernel_launcher(d_inputs, d_outputs, sizes);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float ms = 0.0f;
            cudaEventElapsedTime(&ms, start, stop);
            total_ms += ms;
        }

        return std::make_pair(flops, total_ms / num_runs);
    }
};

template<typename T>
void run_benchmarks(const std::vector<std::unique_ptr<TestCase<T>>>& test_cases,
                   typename BenchmarkRunner<T>::KernelLauncher kernel_launcher) {
    std::cout << std::fixed << std::setprecision(9);
    
    double total_gflops = 0.0;
    int curr_testcase = 0;
    
    for (const auto& test_case : test_cases) {
        BenchmarkRunner<T> runner(*test_case);
        std::pair<size_t, float> benchmark_result = runner.run_benchmark(kernel_launcher, *test_case);
        size_t flops = benchmark_result.first;
        float avg_ms = benchmark_result.second;
        
        double gflops = (flops / 1e9) / (avg_ms / 1000.0);
        total_gflops += gflops;
        
        std::cout << curr_testcase << "," << avg_ms << "," << gflops << std::endl;
        curr_testcase++;
    }
    
    std::cout << (total_gflops / test_cases.size()) << std::endl;
}

int main() {
    auto test_cases = create_test_cases();
    run_benchmarks(test_cases, launch_kernel<float>);
    return 0;
}
