from problem import Problem
import torch
from typing import List, Dict, Any, Tuple
import ctypes


class VectorAdditionProblem(Problem):
    """Vector addition problem."""
    
    def __init__(self):
        super().__init__(
            name="Vector Addition",
            description="Implement a CUDA kernel for vector addition: C = A + B"
        )
    
    def reference_solution(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """PyTorch implementation of vector addition."""
        with torch.no_grad():
            return A + B
    
    def generate_test_cases(self) -> List[Dict[str, Any]]:
        """Generate test cases for vector addition."""
        return [
            {
                "name": "1M elements",
                "dims": (1000000,),
                "create_inputs": lambda: (
                    torch.rand(1000000, device="cuda", dtype=torch.float32),
                    torch.rand(1000000, device="cuda", dtype=torch.float32)
                )
            },

            {
                "name": "5M elements",
                "dims": (5000000,),
                "create_inputs": lambda: (
                    torch.rand(5000000, device="cuda", dtype=torch.float32),
                    torch.rand(5000000, device="cuda", dtype=torch.float32)
                )
            },

            {
                "name": "10M elements",
                "dims": (10000000,),
                "create_inputs": lambda: (
                    torch.rand(10000000, device="cuda", dtype=torch.float32),
                    torch.rand(10000000, device="cuda", dtype=torch.float32)
                )
            },

            {
                "name": "50M elements",
                "dims": (50000000,),
                "create_inputs": lambda: (
                    torch.rand(50000000, device="cuda", dtype=torch.float32),
                    torch.rand(50000000, device="cuda", dtype=torch.float32)
                )
            },
            {
                "name": "100M elements",
                "dims": (100000000,),
                "create_inputs": lambda: (
                    torch.rand(100000000, device="cuda", dtype=torch.float32),
                    torch.rand(100000000, device="cuda", dtype=torch.float32)
                )
            },
            {
                "name": "1B elements",
                "dims": (1000000000,),
                "create_inputs": lambda: (
                    torch.rand(1000000000, device="cuda", dtype=torch.float32),
                    torch.rand(1000000000, device="cuda", dtype=torch.float32)
                )
            }

        ]
    
    def verify_result(self, expected_output: torch.Tensor, actual_output: torch.Tensor) -> Tuple[bool, Dict[str, Any]]:
        """Verify if the vector addition result is correct."""
        is_close = torch.allclose(actual_output, expected_output, rtol=1e-5, atol=1e-5)
        
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
        """Get the function signature for the vector addition solution."""
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # input_a
                ctypes.POINTER(ctypes.c_float),  # input_b
                ctypes.POINTER(ctypes.c_float),  # output
                ctypes.c_size_t                  # N
            ],
            "restype": None
        }
    
    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """Get extra parameters to pass to the CUDA solution."""
        N = test_case["dims"][0]
        return [N]