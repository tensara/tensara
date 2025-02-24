#include "core.hpp"

template<typename T>
class LeakyReLUTest: public TestCase<T> {
public:
    using kernel_func_t = void (*)(T*, T*, size_t, size_t, T);
    
    LeakyReLUTest(size_t n, size_t m, T alpha) {
        this->problem_size_ = n * m;
        this->name_ = std::to_string(n) + "x" + std::to_string(m) + "_alpha" + std::to_string(alpha);
        this->alpha_ = alpha;
        
        auto matrix_shape = std::vector<size_t>{n, m};

        this->inputs_ = {
            std::make_shared<Tensor<T>>(matrix_shape)
        };
        this->outputs_ = {
            std::make_shared<Tensor<T>>(matrix_shape)
        };
    }
    
    void prepare_data(T** host_inputs, T** host_outputs) override {
        const size_t n = this->inputs_[0]->shape()[0];
        const size_t m = this->inputs_[0]->shape()[1];
        
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < m; j++) {
                host_inputs[0][i * m + j] = static_cast<T>(i + j - (n + m)/2);
            }
        }
    }
    
    std::string get_name() const override {
        return this->name_;
    }

    size_t calculate_flops() const override {
        const size_t n = this->inputs_[0]->shape()[0];
        const size_t m = this->inputs_[0]->shape()[1];
        return n * m;
    }

    std::vector<size_t> get_sizes() const override {
        const size_t n = this->inputs_[0]->shape()[0];
        const size_t m = this->inputs_[0]->shape()[1];
        return {n, m};
    }

    void launch_kernel(const std::vector<T*>& inputs, const std::vector<T*>& outputs, 
                      const std::vector<size_t>& sizes, void* kernel_func) override {
        auto typed_func = reinterpret_cast<kernel_func_t>(kernel_func);
        typed_func(inputs[0], outputs[0], sizes[0], sizes[1], alpha_);
    }

private:
    T alpha_;
};

std::vector<std::unique_ptr<TestCase<float>>> create_test_cases() {
    std::vector<std::unique_ptr<TestCase<float>>> test_cases;
    
    std::vector<float> alpha_values = {0.01f, 0.05f, 0.1f, 0.2f};
    std::vector<std::pair<size_t, size_t>> matrix_sizes = {
        {4096, 4096},
        {6144, 4096},
    };

    for (const auto& size : matrix_sizes) {
        for (float alpha : alpha_values) {
            test_cases.push_back(std::make_unique<LeakyReLUTest<float>>(size.first, size.second, alpha));
        }
    }
    
    return test_cases;
}
