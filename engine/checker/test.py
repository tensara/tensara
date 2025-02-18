import torch
import importlib
import sys

def run_test(student_module_name, reference_module_name):
    student_module = importlib.import_module(student_module_name)
    reference_module = importlib.import_module(reference_module_name)
    
    student_vector_add = getattr(student_module, 'cuda_solution')
    intended_vector_add = getattr(reference_module, 'py_solution')
    
    size = 100
    input1 = torch.rand(size, dtype=torch.float32, device='cuda')
    input2 = torch.rand(size, dtype=torch.float32, device='cuda')
    
    student_output = torch.zeros_like(input1)
    intended_output = torch.zeros_like(input1)
    
    student_vector_add(input1, input2, student_output)
    intended_vector_add(input1, input2, intended_output)
    
    if not torch.allclose(student_output, intended_output):
        max_diff = torch.max(torch.abs(student_output - intended_output))
        raise ValueError(f"Test failed! Maximum difference between outputs: {max_diff}")
    
    return "Test passed! Student solution matches intended solution"

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python test.py <student_module_name> <reference_module_name>")
        sys.exit(1)
    
    result = run_test(sys.argv[1], sys.argv[2])
    print(result)