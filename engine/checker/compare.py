import torch
import numpy as np
from typing import Tuple
from .solution import solution as torch_solution
from .cuda_solution import solution as cuda_solution

def generate_random_inputs(size: int, device: str = "cuda") -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate random input tensors for testing.
    
    Args:
        size: Number of elements in each vector
        device: Device to place tensors on
        
    Returns:
        Tuple of (input1, input2) tensors
    """
    input1 = torch.rand(size, device=device)
    input2 = torch.rand(size, device=device)
    return input1, input2

def compare_solutions(size: int = 1000000, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """Compare CUDA and PyTorch implementations for correctness.
    
    Args:
        size: Size of vectors to test
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison
        
    Returns:
        True if implementations match within tolerance
    """
    # Generate random inputs
    input1, input2 = generate_random_inputs(size)
    
    # Get CUDA solution
    cuda_output = torch.empty_like(input1)
    cuda_solution(input1, input2, cuda_output)
    torch.cuda.synchronize()  # Ensure all CUDA operations are complete
    
    # Get PyTorch solution
    torch_output = torch_solution(input1, input2)
    
    # Compare results
    return torch.allclose(cuda_output, torch_output, rtol=rtol, atol=atol)

if __name__ == "__main__":
    # Example usage
    result = compare_solutions()
    print(f"Solutions match: {result}")
