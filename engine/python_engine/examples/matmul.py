
from problem import Problem
import torch
from typing import List, Dict, Any, Tuple
import ctypes


class MatrixMultiplyProblem(Problem):
    """Matrix multiplication problem."""
    
    def __init__(self):
        super().__init__(
            name="Matrix Multiplication",
            description="Implement a CUDA kernel for matrix multiplication: C = A * B"
        )
    
    def reference_solution(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """PyTorch implementation of matrix multiplication."""
        with torch.no_grad():
            return torch.matmul(A, B)
    
    def generate_test_cases(self) -> List[Dict[str, Any]]:
        """Generate test cases for matrix multiplication."""
        return [
            {
                "name": "64x64 identity matrices",
                "dims": (64, 64, 64),
                "create_inputs": lambda: (
                    torch.rand(64, 64, device="cuda", dtype=torch.float32),
                    torch.rand(64, 64, device="cuda", dtype=torch.float32)
                )
            },
            {
                "name": "128x64 random matrices",
                "dims": (128, 64, 32),
                "create_inputs": lambda: (
                    torch.rand(128, 32, device="cuda", dtype=torch.float32),
                    torch.rand(32, 64, device="cuda", dtype=torch.float32)
                )
            },
            {
                "name": "512x512 random matrices",
                "dims": (512, 512, 512),
                "create_inputs": lambda: (
                    torch.rand(512, 512, device="cuda", dtype=torch.float32),
                    torch.rand(512, 512, device="cuda", dtype=torch.float32)
                )
            },
            {
                "name": "2048x2048 random matrices",
                "dims": (2048, 2048, 2048),
                "create_inputs": lambda: (
                    torch.rand(2048, 2048, device="cuda", dtype=torch.float32),
                    torch.rand(2048, 2048, device="cuda", dtype=torch.float32)
                )
            }
        ]
    
    def verify_result(self, expected_output: torch.Tensor, actual_output: torch.Tensor) -> Tuple[bool, Dict[str, Any]]:
        """Verify if the matrix multiplication result is correct."""
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
        """Get the function signature for the matrix multiplication solution."""
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  
                ctypes.POINTER(ctypes.c_float),  
                ctypes.POINTER(ctypes.c_float), 
                ctypes.c_size_t,                 
                ctypes.c_size_t,            
                ctypes.c_size_t          
            ],
            "restype": None
        }

    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """Get extra parameters to pass to the CUDA solution."""
        M, N, K = test_case["dims"]
        return [M, N, K]