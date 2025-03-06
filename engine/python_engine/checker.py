from pathlib import Path
import os
import ctypes
import torch
import tempfile
from typing import Dict, Any
from problem import Problem


class Checker:
    """CUDA solution checker."""
    def __init__(self, problem: Problem):
        self.problem = problem
    
    def check_solution(self, solution_code: str) -> Dict[str, Any]:
        """
        Check a submitted CUDA solution against the reference implementation
        
        Args:
            solution_code: CUDA code for the submitted solution
            
        Returns:
            Dictionary with test results
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            print(f"Checking {self.problem.name} solution...")
            tmpdir_path = Path(tmpdir)
            checker_dir = tmpdir_path / "checker"
            checker_dir.mkdir()
            
            solution_path = checker_dir / "solution.cu"
            solution_path.write_text(solution_code)
            
            makefile_content = """NVCC=nvcc
CFLAGS=-std=c++20 -O2 -Xcompiler -fPIC
SM=75

all: libsolution.so

libsolution.so: solution.cu
\t$(NVCC) $(CFLAGS) -arch=compute_$(SM) -code=sm_$(SM) -shared -o libsolution.so solution.cu

clean:
\trm -f libsolution.so
"""
            
            makefile_path = checker_dir / "Makefile"
            makefile_path.write_text(makefile_content)
            
            print("Compiling solution...")
            os.chdir(checker_dir)
            compile_result = os.system("make 2>&1")
            
            if compile_result != 0:
                print("Compilation failed!")
                compilation_output = os.popen("make 2>&1").read()
                print(compilation_output)
                return {"status": "error", "message": "Compilation failed", "details": compilation_output}
            
            lib_path = os.path.join(checker_dir, "libsolution.so")
            cuda_lib = ctypes.CDLL(lib_path)
            
            func_sig = self.problem.get_function_signature()
            cuda_lib.solution.argtypes = func_sig["argtypes"]
            cuda_lib.solution.restype = func_sig["restype"]
            
            test_cases = self.problem.generate_test_cases()
            
            # Results tracking
            test_results = []
            passed_tests = 0
            total_tests = len(test_cases)
            
            # Run each test case
            for test_id, test_case in enumerate(test_cases, 1):
                test_name = test_case["name"]
                print(f"\nRunning test {test_id}/{total_tests}: {test_name}")
                
                try:
                    # Create input tensors
                    input_tensors = test_case["create_inputs"]()
                    
                    # Prepare output tensor for the CUDA solution
                    actual_output = torch.zeros_like(expected_output)
                    
                    # Get pointers to the GPU memory
                    input_ptrs = [ctypes.cast(tensor.data_ptr(), ctypes.POINTER(ctypes.c_float)) 
                                 for tensor in input_tensors]
                    output_ptr = ctypes.cast(actual_output.data_ptr(), ctypes.POINTER(ctypes.c_float))
                    extra_params = self.problem.get_extra_params(test_case)

                    # Call the CUDA solution
                    cuda_lib.solution(*(input_ptrs + [output_ptr] + extra_params))
                    torch.cuda.synchronize()

                    # Calculate reference result
                    expected_output = self.problem.reference_solution(*input_tensors)
                    
                    # Verify the result
                    is_correct, debug_info = self.problem.verify_result(expected_output, actual_output)
                    
                    if is_correct:
                        status = "PASSED"
                        passed_tests += 1
                        print(f"PASSED")
                    else:
                        status = "FAILED"
                        print(f"FAILED")
                        
                        for key, value in debug_info.items():
                            print(f"  - {key}: {value}")
                    
                except Exception as e:
                    status = "FAILED"
                    debug_info = {"error": str(e)}
                    print(f"❌ FAILED with error: {str(e)}")
                
                test_result = {
                    "test_id": test_id,
                    "name": test_name,
                    "status": status
                }
                
                if status == "FAILED" and "debug_info" in locals():
                    test_result["debug_info"] = debug_info
                
                test_results.append(test_result)
            
            # Print summary
            print("\n" + "="*50)
            print(f"Test Summary for {self.problem.name}:")
            print(f"Passed: {passed_tests}/{total_tests} tests")
            print("="*50)
            
            return {
                "status": "complete",
                "passed_all": passed_tests == total_tests,
                "passed_tests": passed_tests,
                "total_tests": total_tests,
                "test_results": test_results
            }
