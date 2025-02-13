from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="cuda_solution",
    ext_modules=[
        CUDAExtension(
            name="cuda_solution",
            sources=["cuda_bindings.cpp", "solution.cu"],
            extra_compile_args={'cxx': ['-O3'],
                              'nvcc': ['-O3']}
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
) 