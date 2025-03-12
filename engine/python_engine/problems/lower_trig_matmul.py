import torch
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem


class lower_trig_matmul(Problem):
    """Lower triangular matrix multiplication problem."""
    
    def __init__(self):
        super().__init__(
            name="lower-trig-matmul"
        )
    
    def reference_solution(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of lower triangular matrix multiplication.
        
        Args:
            A: First input lower triangular matrix
            B: Second input lower triangular matrix
            
        Returns:
            Result of A * B (also lower triangular)
        """
        with torch.no_grad():
            return torch.tril(torch.matmul(A, B))
    
    def generate_test_cases(self) -> List[Dict[str, Any]]:
        """
        Generate test cases for lower triangular matrix multiplication.
        
        Returns:
            List of test case dictionaries with varying sizes
        """
        # Standard sizes for testing matrix multiplication
        sizes = [
            ("512x512", 512),
            ("1024x1024", 1024),
            ("2048x2048", 2048),
            ("4096x4096", 4096),
            ("8192x8192", 8192)
        ]
        
        return [
            {
                "name": name,
                "dims": (size, size),
                "create_inputs": lambda size=size: (
                    torch.tril(torch.rand(size, size, device="cuda", dtype=torch.float32)),
                    torch.tril(torch.rand(size, size, device="cuda", dtype=torch.float32))
                )
            }
            for name, size in sizes
        ]
    
    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the matrix multiplication result is correct.
        
        Args:
            expected_output: Output from reference solution
            actual_output: Output from submitted solution
            
        Returns:
            Tuple of (is_correct, debug_info)
        """
        is_close = torch.allclose(actual_output, expected_output, rtol=1e-5, atol=1e-5)
        
        debug_info = {}
        if not is_close:
            diff = actual_output - expected_output
            max_diff = torch.max(torch.abs(diff)).item()
            mean_diff = torch.mean(torch.abs(diff)).item()
            
            # Also check if output is lower triangular
            is_lower_triangular = torch.allclose(actual_output, torch.tril(actual_output))
            
            debug_info = {
                "max_difference": max_diff,
                "mean_difference": mean_diff,
                "is_lower_triangular": is_lower_triangular
            }
        
        return is_close, debug_info
    
    def get_function_signature(self) -> Dict[str, Any]:
        """
        Get the function signature for the lower triangular matrix multiplication solution.
        
        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # input_a
                ctypes.POINTER(ctypes.c_float),  # input_b
                ctypes.POINTER(ctypes.c_float),  # output
                ctypes.c_size_t                  # N (matrix dimension)
            ],
            "restype": None
        }
    
    def get_flops(self, test_case: Dict[str, Any]) -> int:
        """
        Get the number of floating point operations for the problem.

        IMPORTANT: Comments are required. Outline the FLOPs calculation.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            Number of floating point operations
        """
        # For an NxN triangular matrix, we have N(N+1)/2 non-zero elements
        # For each element in the result matrix, we need to compute a dot product
        # of corresponding row and column elements 
        N = test_case["dims"][0]
        
        # Number of elements in triangular matrix
        num_elements = (N * (N + 1)) // 2
        
        # For lower triangular matrices, each element in the result 
        # requires fewer multiplications than a full matrix multiplication
        # This is a conservative estimate that counts 2 FLOPs (mul + add) per element
        total_flops = 0
        for i in range(N):
            for j in range(i + 1):  # Only lower triangle
                # For each element C[i,j], we compute sum(A[i,k] * B[k,j]) for k from 0 to min(i,j)
                dot_product_length = j + 1  # Only need to go up to j+1 terms
                total_flops += 2 * dot_product_length  # mul + add for each term
        
        return total_flops
    
    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            List containing the matrix dimension N
        """
        N = test_case["dims"][0]
        return [N]