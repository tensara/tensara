#include "core.hpp"
#include <random>

template<typename T>
class VectorAddTest: public TestCase<T> {
public:
    using kernel_func_t = void (*)(T*, T*, T*, size_t);
    
    explicit VectorAddTest(size_t n, unsigned int seed = 42) : rng_(seed) {
        this->problem_size_ = n;
        this->name_ = "n = " + std::to_string(n);
        
        auto vec_shape = std::vector<size_t>{n};
        this->inputs_ = {
            std::make_shared<Tensor<T>>(vec_shape),
            std::make_shared<Tensor<T>>(vec_shape)
        };
        this->outputs_ = {
            std::make_shared<Tensor<T>>(vec_shape)
        };
    }
    
    void prepare_data(T** host_inputs, T** host_outputs) override {
        const size_t n = this->problem_size_;
        
        std::uniform_real_distribution<T> dist(-100.0, 100.0);
        
        for (size_t i = 0; i < n; i++) {
            host_inputs[0][i] = dist(rng_);
            host_inputs[1][i] = dist(rng_);
        }
    }
    
    size_t calculate_flops() const override {
        return this->problem_size_;
    }
    
    std::string get_name() const override {
        return this->name_;
    }

    std::vector<size_t> get_sizes() const override {
        return {this->problem_size_};
    }

    void launch_kernel(const std::vector<T*>& inputs, const std::vector<T*>& outputs, 
                      const std::vector<size_t>& sizes, void* kernel_func) override {
        auto typed_func = reinterpret_cast<kernel_func_t>(kernel_func);
        typed_func(inputs[0], inputs[1], outputs[0], sizes[0]);
    }

private:
    std::string name_;
    std::mt19937 rng_;
};

std::vector<std::unique_ptr<TestCase<float>>> create_test_cases() {
    std::vector<std::unique_ptr<TestCase<float>>> test_cases;
    unsigned int base_seed = 98765;
    test_cases.push_back(std::make_unique<VectorAddTest<float>>(1000000, base_seed++));
    test_cases.push_back(std::make_unique<VectorAddTest<float>>(5000000, base_seed++));
    test_cases.push_back(std::make_unique<VectorAddTest<float>>(10000000, base_seed++));
    test_cases.push_back(std::make_unique<VectorAddTest<float>>(50000000, base_seed++));
    test_cases.push_back(std::make_unique<VectorAddTest<float>>(100000000, base_seed++));
    test_cases.push_back(std::make_unique<VectorAddTest<float>>(500000000, base_seed++));
    test_cases.push_back(std::make_unique<VectorAddTest<float>>(1000000000, base_seed++));
    return test_cases;
} 
