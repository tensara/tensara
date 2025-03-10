from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any, Optional


class Problem(ABC):
    """Base class for defining CUDA problems."""
    
    def __init__(self, name: str, description: str):
        """
        Initialize a CUDA problem.
        
        Args:
            name: Name of the problem
            description: Problem description
        """
        self.name = name
        self.description = description
    
    @abstractmethod
    def reference_solution(self, *args, **kwargs) -> Any:
        """
        Reference implementation using PyTorch.
        """
        pass
    
    @abstractmethod
    def generate_test_cases(self) -> List[Dict[str, Any]]:
        """
        Generate test cases for this problem.
        
        Returns:
            List of test case dictionaries
        """
        pass
    
    @abstractmethod
    def verify_result(self, expected_output: Any, actual_output: Any) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the actual output matches the expected output.
        
        Args:
            expected_output: Output from reference solution
            actual_output: Output from submitted solution
            
        Returns:
            Tuple of (is_correct, debug_info)
        """
        pass
    
    @abstractmethod
    def get_function_signature(self) -> Dict[str, Any]:
        """
        Get the function signature for the CUDA solution.
        
        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        pass
    
    @abstractmethod
    def get_flops(self, test_case: Dict[str, Any]) -> int:
        """
        Get the number of floating point operations for the problem.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            Number of floating point operations
        """
        pass
    
    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution for a specific test case.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            List of extra parameters
        """
        return []