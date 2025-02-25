#include "core.hpp"
#include <random>

template<typename T>
class MatrixScalarTest: public TestCase<T> {
public:
    using kernel_func_t = void (*)(T*, T, T*, size_t, size_t);
    
    MatrixScalarTest(size_t m, size_t n, T scalar, unsigned int seed = 42) : rng_(seed) {
        this->problem_size_ = m * n;
        this->name_ = std::to_string(m) + "x" + std::to_string(n) + "_scalar" + std::to_string(scalar);
        this->scalar_ = scalar;
        
        auto matrix_shape = std::vector<size_t>{m, n};

        this->inputs_ = {
            std::make_shared<Tensor<T>>(matrix_shape),
            std::make_shared<Tensor<T>>(std::vector<size_t>{1})
        };
        this->outputs_ = {
            std::make_shared<Tensor<T>>(matrix_shape)
        };
    }
    
    void prepare_data(T** host_inputs, T** host_outputs) override {
        const size_t m = this->inputs_[0]->shape()[0];
        const size_t n = this->inputs_[0]->shape()[1];
        
        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < n; j++) {
                host_inputs[0][i * n + j] = dist(rng_);
            }
        }
        
        host_inputs[1][0] = this->scalar_;
    }
    
    size_t calculate_flops() const override {
        const size_t m = this->inputs_[0]->shape()[0];
        const size_t n = this->inputs_[0]->shape()[1];
        return m * n;
    }

    std::vector<size_t> get_sizes() const override {
        const size_t m = this->inputs_[0]->shape()[0];
        const size_t n = this->inputs_[0]->shape()[1];
        return {n, m};
    }

    void launch_kernel(const std::vector<T*>& inputs, const std::vector<T*>& outputs, 
                      const std::vector<size_t>& sizes, void* kernel_func) override {
        auto typed_func = reinterpret_cast<kernel_func_t>(kernel_func);
        typed_func(inputs[0], inputs[1][0], outputs[0], sizes[0], sizes[1]);
    }

private:
    std::mt19937 rng_;
    T scalar_;
};

std::vector<std::unique_ptr<TestCase<float>>> create_test_cases() {
    std::vector<std::unique_ptr<TestCase<float>>> test_cases;
    
    std::vector<float> scalar_values = {0.5f, 2.0f, -1.0f, 10.0f};
    std::vector<size_t> matrix_sizes = {4096, 6144, 7168, 8192, 9216};
    
    for (size_t size: matrix_sizes) {
        for (float scalar: scalar_values) {
            test_cases.push_back(std::make_unique<MatrixScalarTest<float>>(size, 4096, scalar, 123));
            test_cases.push_back(std::make_unique<MatrixScalarTest<float>>(size, 8192, scalar, 3928));
        }
    }
    
    return test_cases;
}
