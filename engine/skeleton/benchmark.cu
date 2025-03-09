#include "solution.cu"

#include "core.hpp"
#include "tests.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

static const size_t MINIMUM_RUNS = 1;
static const double MINIMUM_TIME_SECS = 0.5;

static const size_t MINIMUM_WARMUP_RUNS = 1;
static const float MINIMUM_WARMUP_TIME_SECS = 0.5;

size_t prefix_sum(const std::vector<float> &v, float target) {
    // return first index where sum[0:i] >= target
    size_t i = 0;
    float sum = 0.0;

    while (i < v.size() && sum < target) {
        sum += v[i];
        i++;
    }

    return i;
}

template <typename T> class BenchmarkRunner {
  private:
    cudaEvent_t start, stop;
    std::vector<T *> h_inputs;
    std::vector<T *> h_outputs;
    std::vector<const T *> d_inputs;
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
            cudaMalloc(const_cast<T**>(&d_inputs[i]), size * sizeof(T));
            cudaMemcpy(const_cast<T*>(d_inputs[i]), h_inputs[i], size * sizeof(T), cudaMemcpyHostToDevice);
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
            cudaFree(const_cast<T*>(ptr));
        for (auto ptr : d_outputs)
            cudaFree(ptr);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

  public:
    typedef void (*KernelLauncher)(const std::vector<T *> &, const std::vector<T *> &,
                                   const std::vector<size_t> &);

    std::pair<size_t, float> run_benchmark(TestCase<T> &test_case) {
        size_t flops = test_case.calculate_flops();
        std::vector<size_t> sizes = test_case.get_sizes();

        double elapsed = 0.0;
        std::vector<float> runtimes;

        while (elapsed < (MINIMUM_TIME_SECS + MINIMUM_WARMUP_TIME_SECS) ||
               runtimes.size() < (MINIMUM_RUNS + MINIMUM_WARMUP_RUNS)) {
            cudaEventRecord(start);
            test_case.launch_kernel(d_inputs, d_outputs, sizes, reinterpret_cast<void *>(solution));
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaDeviceSynchronize();

            float ms = 1e+4;
            cudaEventElapsedTime(&ms, start, stop);
            runtimes.push_back(ms);

            elapsed += ms / 1000.0;
        }

        size_t warmup_index = prefix_sum(runtimes, MINIMUM_WARMUP_TIME_SECS * 1000);
        warmup_index = std::max(warmup_index, MINIMUM_WARMUP_RUNS);

        // cut off warmup runs
        runtimes.erase(runtimes.begin(), runtimes.begin() + warmup_index);

        const float mean_time =
            std::accumulate(runtimes.begin(), runtimes.end(), 0.0) / runtimes.size();

        return std::make_pair(flops, mean_time);
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

        std::cout << curr_testcase << "," << test_case->get_name() << "," << avg_ms << "," << gflops
                  << std::endl;
        curr_testcase++;
    }

    std::cout << (total_gflops / test_cases.size()) << std::endl;
}

int main() {
    auto test_cases = create_test_cases();
    run_benchmarks(test_cases);
    return 0;
}
