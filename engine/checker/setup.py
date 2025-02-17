from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

def build_cuda_module(module_name, sources):
    setup(
        name=module_name,
        ext_modules=[
            CUDAExtension(
                name=module_name,
                sources=sources,
                extra_compile_args={'cxx': ['-O3'],
                                  'nvcc': ['-O3']}
            ),
        ],
        cmdclass={
            'build_ext': BuildExtension
        }
    )

if __name__ == "__main__":
    # Default behavior for backward compatibility
    build_cuda_module("cuda_solution", ["cuda_bindings.cpp", "solution.cu"]) 