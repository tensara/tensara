from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any
import ctypes
import hashlib
import torch

TYPE_TO_CTYPE = {
    "float": ctypes.c_float,
    "double": ctypes.c_double,
    "int": ctypes.c_int,
    "size_t": ctypes.c_size_t,
    "uint32_t": ctypes.c_uint32,
    "uint64_t": ctypes.c_uint64,
}

TYPE_TO_TORCH_DTYPE = {
    "float": torch.float32,
    "double": torch.float64,
    "int": torch.int32,
    "size_t": torch.int64,
    "uint32_t": torch.int32,
    "uint64_t": torch.int64,
}

class Problem(ABC):
    """Base class for defining problems."""

    parameters = None

    def __init__(self, name: str, time_limit: int = 100):
        """
        Initialize a problem.

        Args:
            name: Name of the problem
        """
        self.name = name
        self.time_limit = time_limit

    def param_dtype(self, key: str | int) -> torch.dtype:
        """Return the torch dtype for a parameter by name or index."""
        if not self.parameters:
            raise ValueError("Problem has no parameters defined")
        if isinstance(key, int):
            return TYPE_TO_TORCH_DTYPE.get(self.parameters[key]["type"], torch.float32)
        for p in self.parameters:
            if p["name"] == key:
                return TYPE_TO_TORCH_DTYPE.get(p["type"], torch.float32)
        raise KeyError(f"No parameter named '{key}'")

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
        """
        pass

    @abstractmethod
    def generate_sample(self) -> List[Dict[str, Any]]:
        """
        Generate a sample test case for this problem.
        """
        pass

    @abstractmethod
    def verify_result(
        self, expected_output: torch.Tensor, actual_output: torch.Tensor
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the actual output matches the expected output.

        Args:
            expected_output: Output from reference solution
            actual_output: Output from submitted solution

        Returns:
            Tuple of (is_correct, debug_info)
        """
        pass

    def get_function_signature(self) -> Dict[str, Any]:
        """
        Get the function signature for the CUDA solution.

        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        if self.parameters:
            argtypes = []
            for p in self.parameters:
                ctype = TYPE_TO_CTYPE[p["type"]]
                if p.get("pointer"):
                    ctype = ctypes.POINTER(ctype)
                argtypes.append(ctype)
            return {"argtypes": argtypes, "restype": None}
        raise NotImplementedError(
            "Subclass must define `parameters` or override `get_function_signature()`"
        )

    def get_flops(self, test_case: Dict[str, Any]) -> int:
        """
        Get the number of floating point operations for the problem.

        Args:
            test_case: The test case dictionary

        Returns:
            Number of floating point operations
        """
        return None

    def supports_flops(self) -> bool:
        return self.get_flops is not Problem.get_flops

    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution for a specific test case.

        Args:
            test_case: The test case dictionary

        Returns:
            List of extra parameters
        """
        return []

    @staticmethod
    def get_seed(seed_str: str) -> int:
        """
        Generate deterministic seed from a string.
        Uses MD5 for deterministic hashing across Python processes.
        """
        hash_bytes = hashlib.md5(seed_str.encode()).digest()
        seed = int.from_bytes(hash_bytes[:4], "little") & 0x7FFFFFFF
        return seed
