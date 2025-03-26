#include "core.hpp"

template<typename T>
class MatrixMultiplyTest: public TestCase<T> {
public:
    using kernel_func_t = void (*)(T*, T*, T*, size_t, size_t, size_t);
    
    MatrixMultiplyTest(size_t m, size_t n, size_t k) {
        this->problem_size_ = m * n * k;
        this->name_ = std::to_string(m) + "x" + std::to_string(k) + " x " + std::to_string(k) + "x" + std::to_string(n);
        
        auto matrix_a_shape = std::vector<size_t>{m, k};
        auto matrix_b_shape = std::vector<size_t>{k, n}; 
        auto matrix_c_shape = std::vector<size_t>{m, n};

        this->inputs_ = {
            std::make_shared<Tensor<T>>(matrix_a_shape),
            std::make_shared<Tensor<T>>(matrix_b_shape)
        };
        this->outputs_ = {
            std::make_shared<Tensor<T>>(matrix_c_shape)
        };
    }
    
    void prepare_data(T** host_inputs, T** host_outputs) override {
        const size_t m = this->inputs_[0]->shape()[0];
        const size_t k = this->inputs_[0]->shape()[1];
        const size_t n = this->inputs_[1]->shape()[1];
        
        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < k; j++) {
                host_inputs[0][i * k + j] = static_cast<T>(i + j);
            }
        }
        
        for (size_t i = 0; i < k; i++) {
            for (size_t j = 0; j < n; j++) {
                host_inputs[1][i * n + j] = static_cast<T>(i * j);
            }
        }
    }
    
    std::string get_name() const override {
        return this->name_;
    }

    size_t calculate_flops() const override {
        const size_t m = this->inputs_[0]->shape()[0];
        const size_t k = this->inputs_[0]->shape()[1];
        const size_t n = this->inputs_[1]->shape()[1];
        return m * n * k * 2;
    }

    std::vector<size_t> get_sizes() const override {
        const size_t m = this->inputs_[0]->shape()[0];
        const size_t k = this->inputs_[0]->shape()[1];
        const size_t n = this->inputs_[1]->shape()[1];
        return {m, n, k};
    }

    void launch_kernel(const std::vector<T*>& inputs, const std::vector<T*>& outputs, 
                      const std::vector<size_t>& sizes, void* kernel_func) override {
        auto typed_func = reinterpret_cast<kernel_func_t>(kernel_func);
        typed_func(inputs[0], inputs[1], outputs[0], sizes[0], sizes[1], sizes[2]);
    }
};

std::vector<std::unique_ptr<TestCase<float>>> create_test_cases() {
    std::vector<std::unique_ptr<TestCase<float>>> test_cases;
    
    test_cases.push_back(std::make_unique<MatrixMultiplyTest<float>>(4092, 4092, 4092));
    test_cases.push_back(std::make_unique<MatrixMultiplyTest<float>>(8192, 4092, 8192));
    test_cases.push_back(std::make_unique<MatrixMultiplyTest<float>>(4092, 8192, 4092));
    test_cases.push_back(std::make_unique<MatrixMultiplyTest<float>>(8192, 8192, 8192));

    return test_cases;
}
