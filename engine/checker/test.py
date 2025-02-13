import torch
from cuda_solution import vector_add

# Create sample data on GPU
size = 1000000
input1 = torch.rand(size, dtype=torch.float32, device='cuda')
input2 = torch.rand(size, dtype=torch.float32, device='cuda')
output = torch.zeros_like(input1)

# Call CUDA kernel
vector_add(input1, input2, output)

# Verify result
expected = input1 + input2
assert torch.allclose(output, expected)
print("Test passed! CUDA kernel matches PyTorch addition")