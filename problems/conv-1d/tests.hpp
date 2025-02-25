#include "core.hpp"
#include <random>

template<typename T>
class Conv1DTest: public TestCase<T> {
public:
    using kernel_func_t = void (*)(T*, T*, T*, size_t, size_t);
    
    Conv1DTest(size_t n, size_t k, unsigned int seed = 42) : rng_(seed) {
        this->problem_size_ = n;
        this->name_ = "N" + std::to_string(n) + "_K" + std::to_string(k);
        
        auto input_shape = std::vector<size_t>{n};
        auto kernel_shape = std::vector<size_t>{k};
        auto output_shape = std::vector<size_t>{n};

        this->inputs_ = {
            std::make_shared<Tensor<T>>(input_shape),
            std::make_shared<Tensor<T>>(kernel_shape)
        };
        this->outputs_ = {
            std::make_shared<Tensor<T>>(output_shape)
        };
    }
    
    void prepare_data(T** host_inputs, T** host_outputs) override {
        const size_t N = this->inputs_[0]->shape()[0];
        const size_t K = this->inputs_[1]->shape()[0];
        
        std::uniform_real_distribution<T> input_dist(-10.0, 10.0);
        std::normal_distribution<T> kernel_dist(0.0, 1.0);
        
        for (size_t i = 0; i < N; i++) {
            host_inputs[0][i] = input_dist(rng_);
        }

        for (size_t i = 0; i < K; i++) {
            host_inputs[1][i] = kernel_dist(rng_);
        }
    }

    size_t calculate_flops() const override {
        const size_t N = this->inputs_[0]->shape()[0];
        const size_t K = this->inputs_[1]->shape()[0];
        return 2 * N * K;
    }

    std::vector<size_t> get_sizes() const override {
        const size_t N = this->inputs_[0]->shape()[0];
        const size_t K = this->inputs_[1]->shape()[0];
        return {N, K};
    }

    void launch_kernel(const std::vector<T*>& inputs, const std::vector<T*>& outputs, 
                      const std::vector<size_t>& sizes, void* kernel_func) override {
        auto typed_func = reinterpret_cast<kernel_func_t>(kernel_func);
        typed_func(inputs[0], inputs[1], outputs[0], sizes[0], sizes[1]);
    }

private:
    std::mt19937 rng_;
};

std::vector<std::unique_ptr<TestCase<float>>> create_test_cases() {
    std::vector<std::unique_ptr<TestCase<float>>> test_cases;
    
    std::vector<std::pair<size_t, size_t>> sizes = {
        {65536, 8192},
        {32768, 8192}, 
        {131072, 8192},
        {524288, 8192},
    };

    unsigned int base_seed = 42;
    for (const auto& [N, K] : sizes) {
        test_cases.push_back(std::make_unique<Conv1DTest<float>>(N, K, base_seed++));
    }
    
    return test_cases;
}
