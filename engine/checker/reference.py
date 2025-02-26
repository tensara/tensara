import torch
import numpy as np
import checker_bindings

def matrix_multiply_reference(input_a, input_b, m, n, k):
    """
    PyTorch implementation of matrix multiplication.
    
    Args:
        input_a: numpy array of shape (m, k)
        input_b: numpy array of shape (k, n)
        m, n, k: dimensions
        
    Returns:
        numpy array of shape (m, n) containing the result
    """
    # Convert numpy arrays to PyTorch tensors
    tensor_a = torch.from_numpy(np.asarray(input_a)).float()
    tensor_b = torch.from_numpy(np.asarray(input_b)).float()
    
    # Move tensors to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tensor_a = tensor_a.to(device)
    tensor_b = tensor_b.to(device)
    
    # Perform matrix multiplication
    result = torch.matmul(tensor_a, tensor_b)
    
    # Convert result back to numpy array on CPU
    return result.cpu().numpy()

def main():
    # Register our PyTorch reference implementation
    checker_bindings.register_reference(matrix_multiply_reference)
    
    # Run the checker
    try:
        checker_bindings.run_checker()
        print("All tests passed!")
    except RuntimeError as e:
        print(f"Checker failed: {e}")

if __name__ == "__main__":
    main() 