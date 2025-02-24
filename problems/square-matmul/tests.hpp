#include "core.hpp"

template<typename T>
class SquareMatmulTest: public TestCase<T> {
public:
    using kernel_func_t = void (*)(T*, T*, T*, size_t);
    
    SquareMatmulTest(size_t n) {
        this->problem_size_ = n * n * n;
        this->name_ = std::to_string(n) + "x" + std::to_string(n);
        
        auto matrix_shape = std::vector<size_t>{n, n};

        this->inputs_ = {
            std::make_shared<Tensor<T>>(matrix_shape),
            std::make_shared<Tensor<T>>(matrix_shape)
        };
        this->outputs_ = {
            std::make_shared<Tensor<T>>(matrix_shape)
        };
    }
    
    void prepare_data(T** host_inputs, T** host_outputs) override {
        const size_t n = this->inputs_[0]->shape()[0];
        
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < n; j++) {
                host_inputs[0][i * n + j] = static_cast<T>(i + j);
                host_inputs[1][i * n + j] = static_cast<T>(i * j);
            }
        }
    }
    
    std::string get_name() const override {
        return this->name_;
    }

    size_t calculate_flops() const override {
        const size_t n = this->inputs_[0]->shape()[0];
        return n * n * n * 2;
    }

    std::vector<size_t> get_sizes() const override {
        const size_t n = this->inputs_[0]->shape()[0];
        return {n};
    }

    void launch_kernel(const std::vector<T*>& inputs, const std::vector<T*>& outputs, 
                      const std::vector<size_t>& sizes, void* kernel_func) override {
        auto typed_func = reinterpret_cast<kernel_func_t>(kernel_func);
        typed_func(inputs[0], inputs[1], outputs[0], sizes[0]);
    }
};

std::vector<std::unique_ptr<TestCase<float>>> create_test_cases() {
    std::vector<std::unique_ptr<TestCase<float>>> test_cases;
    
    test_cases.push_back(std::make_unique<SquareMatmulTest<float>>(4096));
    test_cases.push_back(std::make_unique<SquareMatmulTest<float>>(6144));
    test_cases.push_back(std::make_unique<SquareMatmulTest<float>>(7168));
    test_cases.push_back(std::make_unique<SquareMatmulTest<float>>(8192));
    test_cases.push_back(std::make_unique<SquareMatmulTest<float>>(9216));
    return test_cases;
}
