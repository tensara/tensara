import torch

def py_solution(input1, input2, output):
    output.copy_(input1 + input2) 