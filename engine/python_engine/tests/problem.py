import torch
import ctypes
from typing import List, Dict, Tuple, Any

from problem import Problem


class matrix_multiplication(Problem):
    """Matrix multiplication problem."""
    
    def __init__(self):
        super().__init__(
            name="matrix-multiplication"
        )
    
    def reference_solution(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of matrix multiplication.
        
        Args:
            A: First input matrix
            B: Second input matrix
            
        Returns:
            Result of A * B
        """
        with torch.no_grad():
            return torch.matmul(A, B)
    
    def generate_test_cases(self) -> List[Dict[str, Any]]:
        """
        Generate test cases for matrix multiplication.
        
        Returns:
            List of test case dictionaries with varying matrix dimensions
        """
        # Matrix dimensions: (M, K) × (K, N) = (M, N)
        # dims represents (M, N, K)
        test_matrices = [
            {
                "name": "4092x4092 x 4092x4092 matrices",
                "dims": (4092, 4092, 4092),
            },
            {
                "name": "8192x8192 x 8192x4092 matrices",
                "dims": (8192, 4092, 8192),
            },
            {
                "name": "4092x4092 x 4092x8192 matrices",
                "dims": (4092, 8192, 4092),
            },
            {
                "name": "8192x8192 x 8192x8192 matrices",
                "dims": (8192, 8192, 8192),
            }
        ]
        
        return [
            {
                "name": matrix["name"],
                "dims": matrix["dims"],
                "create_inputs": lambda m=matrix["dims"]: (
                    torch.rand(m[0], m[2], device="cuda", dtype=torch.float32),
                    torch.rand(m[2], m[1], device="cuda", dtype=torch.float32)
                )
            }
            for matrix in test_matrices
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
        is_close = torch.allclose(actual_output, expected_output, rtol=1e-4, atol=1e-4)
        
        debug_info = {}
        if not is_close:
            diff = actual_output - expected_output
            max_diff = torch.max(torch.abs(diff)).item()
            mean_diff = torch.mean(torch.abs(diff)).item()
            
            debug_info = {
                "max_difference": max_diff,
                "mean_difference": mean_diff
            }
        
        return is_close, debug_info
    
    def get_function_signature(self) -> Dict[str, Any]:
        """
        Get the function signature for the matrix multiplication solution.
        
        IMPORTANT: Comments are required. Outline the FLOPs calculation.
        
        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # matrix_a
                ctypes.POINTER(ctypes.c_float),  # matrix_b
                ctypes.POINTER(ctypes.c_float),  # matrix_c (output)
                ctypes.c_size_t,                 # M (rows in A and C)
                ctypes.c_size_t,                 # N (columns in B and C)
                ctypes.c_size_t                  # K (columns in A, rows in B)
            ],
            "restype": None
        }
    
    def get_flops(self, test_case: Dict[str, Any]) -> int:
        """
        Get the number of floating point operations for the problem.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            Number of floating point operations
        """
        # Matrix multiplication FLOPS = 2 * M * N * K
        # (One multiply and one add for each cell in the result, done K times)
        M, N, K = test_case["dims"]
        return 2 * M * N * K
    
    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            List containing the dimensions M, N, K
        """
        M, N, K = test_case["dims"]
        return [M, N, K]