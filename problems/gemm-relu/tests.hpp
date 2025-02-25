#include "core.hpp"

template<typename T>
class GEMMReLUTest: public TestCase<T> {
public:
    using kernel_func_t = void (*)(T*, T*, T*, T*, size_t, size_t, size_t);
    
    GEMMReLUTest(size_t batch_size, size_t in_features, size_t out_features) {
        this->problem_size_ = batch_size * in_features * out_features;
        this->name_ = std::to_string(batch_size) + "x" + std::to_string(in_features) + "x" + std::to_string(out_features);
        
        auto input_shape = std::vector<size_t>{batch_size, in_features};
        auto weight_shape = std::vector<size_t>{out_features, in_features};
        auto bias_shape = std::vector<size_t>{out_features};
        auto output_shape = std::vector<size_t>{batch_size, out_features};

        this->inputs_ = {
            std::make_shared<Tensor<T>>(input_shape),
            std::make_shared<Tensor<T>>(weight_shape),
            std::make_shared<Tensor<T>>(bias_shape)
        };
        this->outputs_ = {
            std::make_shared<Tensor<T>>(output_shape)
        };
    }
    
    void prepare_data(T** host_inputs, T** host_outputs) override {
        const size_t B = this->inputs_[0]->shape()[0];
        const size_t N = this->inputs_[0]->shape()[1];
        const size_t M = this->outputs_[0]->shape()[1];
        
        for (size_t i = 0; i < B; i++) {
            for (size_t j = 0; j < N; j++) {
                host_inputs[0][i * N + j] = static_cast<T>((i + j) % 7 - 3) * 0.1f;
            }
        }

        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                host_inputs[1][i * N + j] = static_cast<T>((i - j) % 5 - 2) * 0.1f;
            }
        }

        for (size_t i = 0; i < M; i++) {
            host_inputs[2][i] = static_cast<T>(i % 3 - 1) * 0.1f;
        }
    }
    
    std::string get_name() const override {
        return this->name_;
    }

    size_t calculate_flops() const override {
        const size_t B = this->inputs_[0]->shape()[0];
        const size_t N = this->inputs_[0]->shape()[1];
        const size_t M = this->outputs_[0]->shape()[1];
        return 2 * B * N * M + B * M;
    }

    std::vector<size_t> get_sizes() const override {
        const size_t B = this->inputs_[0]->shape()[0];
        const size_t N = this->inputs_[0]->shape()[1];
        const size_t M = this->outputs_[0]->shape()[1];
        return {B, N, M};
    }

    void launch_kernel(const std::vector<T*>& inputs, const std::vector<T*>& outputs, 
                      const std::vector<size_t>& sizes, void* kernel_func) override {
        auto typed_func = reinterpret_cast<kernel_func_t>(kernel_func);
        typed_func(inputs[0], inputs[1], inputs[2], outputs[0], sizes[0], sizes[1], sizes[2]);
    }

private:
    std::string name_;
};

std::vector<std::unique_ptr<TestCase<float>>> create_test_cases() {
    std::vector<std::unique_ptr<TestCase<float>>> test_cases;
    
    std::vector<std::tuple<size_t, size_t, size_t>> sizes = {
        {1024, 1024, 1024},
        {2048, 1024, 1024},
        {4096, 1024, 1024},
        {6144, 1024, 1024}
    };

    for (const auto& [B, N, M] : sizes) {
        test_cases.push_back(std::make_unique<GEMMReLUTest<float>>(B, N, M));
    }
    
    return test_cases;
}
