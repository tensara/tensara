#pragma once
#include <vector>
#include <memory>
#include <numeric>
#include <functional>
#include <string>

template<typename T>
class Tensor {
public:
    Tensor(const std::vector<size_t>& shape) 
        : shape_(shape) {
        size_ = std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<size_t>());
    }

    const std::vector<size_t>& shape() const { return shape_; }
    size_t size() const { return size_; }
    size_t num_dimensions() const { return shape_.size(); }

    size_t dimension(size_t idx) const { 
        return idx < shape_.size() ? shape_[idx] : 1; 
    }

private:
    std::vector<size_t> shape_;
    size_t size_;
};

template<typename T>
class TestCase {
public:
    TestCase() = default;
    virtual ~TestCase() = default;

    TestCase(const TestCase&) = delete;
    TestCase& operator=(const TestCase&) = delete;
    TestCase(TestCase&&) = default;
    TestCase& operator=(TestCase&&) = default;

    const std::vector<std::shared_ptr<Tensor<const T>>>& input_shapes() const { return inputs_; }
    const std::vector<std::shared_ptr<Tensor<T>>>& output_shapes() const { return outputs_; }
    
    size_t problem_size() const { return problem_size_; }
    
    virtual std::string get_name() const { return name_; }
    virtual std::vector<size_t> get_sizes() const { return {problem_size_}; }
    virtual void launch_kernel(const std::vector<const T*>& inputs, const std::vector<T*>& outputs, 
                             const std::vector<size_t>& sizes, void* kernel_func) = 0;

    virtual size_t calculate_flops() const = 0;
    virtual void prepare_data(T** host_inputs, T** host_outputs) = 0;

protected:
    std::vector<std::shared_ptr<Tensor<const T>>> inputs_;
    std::vector<std::shared_ptr<Tensor<T>>> outputs_;
    size_t problem_size_ = 0;
    std::string name_;
}; 
