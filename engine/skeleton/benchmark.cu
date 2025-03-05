#include "solution.cu"

#include "core.hpp"
#include "tests.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

static const size_t WARMUP_RUNS = 10;
static const size_t MINIMUM_RUNS = 20;
static const double MINIMUM_TIME_SECS = 1.0;

inline float median(std::vector<float> &v) {
    // return median of v
    std::sort(v.begin(), v.end());
    if (v.size() % 2 == 0) {
        return (v[v.size() / 2 - 1] + v[v.size() / 2]) / 2;
    } else {
        return v[v.size() / 2];
    }
}

template <typename T>
class BenchmarkRunner {
  private:
    cudaEvent_t start, stop;
    std::vector<T *> h_inputs;
    std::vector<T *> h_outputs;
    std::vector<T *> d_inputs;
    std::vector<T *> d_outputs;

  public:
    BenchmarkRunner(TestCase<T> &test_case) {

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
        for (auto ptr : h_inputs)
            delete[] ptr;
        for (auto ptr : h_outputs)
            delete[] ptr;

        for (auto ptr : d_inputs)
            cudaFree(ptr);
        for (auto ptr : d_outputs)
            cudaFree(ptr);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

  public:
    typedef void (*KernelLauncher)(const std::vector<T *> &, const std::vector<T *> &, const std::vector<size_t> &);

    std::pair<size_t, float> run_benchmark(TestCase<T> &test_case) {
        size_t flops = test_case.calculate_flops();
        std::vector<size_t> sizes = test_case.get_sizes();

        test_case.launch_kernel(d_inputs, d_outputs, sizes, reinterpret_cast<void *>(solution));
        cudaDeviceSynchronize();

        auto start_time = std::chrono::high_resolution_clock::now();
        double elapsed = 0.0;

        std::vector<float> runtimes;

        for (size_t i = 0; i < WARMUP_RUNS; i++) {
            test_case.launch_kernel(d_inputs, d_outputs, sizes, reinterpret_cast<void *>(solution));
            cudaDeviceSynchronize();
        }

        while (elapsed < MINIMUM_TIME_SECS) {
            cudaEventRecord(start);
            test_case.launch_kernel(d_inputs, d_outputs, sizes, reinterpret_cast<void *>(solution));
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaDeviceSynchronize();

            float ms = 1e+4;
            cudaEventElapsedTime(&ms, start, stop);
            runtimes.push_back(ms);

            elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
        }

        return std::make_pair(flops, median(runtimes));
    }
};

template <typename T>
void run_benchmarks(const std::vector<std::unique_ptr<TestCase<T>>> &test_cases) {
    std::cout << std::fixed << std::setprecision(9);

    double total_gflops = 0.0;
    int curr_testcase = 1;

    for (const auto &test_case : test_cases) {
        BenchmarkRunner<T> runner(*test_case);
        std::pair<size_t, float> benchmark_result = runner.run_benchmark(*test_case);
        size_t flops = benchmark_result.first;
        float avg_ms = benchmark_result.second;

        double gflops = (flops / 1e9) / (avg_ms / 1000.0);
        total_gflops += gflops;

        std::cout << curr_testcase << "," << test_case->get_name() << "," << avg_ms << "," << gflops << std::endl;
        curr_testcase++;
    }

    std::cout << (total_gflops / test_cases.size()) << std::endl;
}

int main() {
    auto test_cases = create_test_cases();
    run_benchmarks(test_cases);
    return 0;
}
