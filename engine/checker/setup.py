from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
import subprocess

# A CMakeExtension to build our module using the Makefile
class MakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class MakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)
    
    def build_extension(self, ext):
        # Build using make
        subprocess.check_call(['make', 'all'], cwd=ext.sourcedir)
        
        # Copy the extension file to the appropriate location
        ext_path = self.get_ext_fullpath(ext.name)
        os.makedirs(os.path.dirname(ext_path), exist_ok=True)
        
        # Get the actual extension filename from the Makefile output
        ext_file = f"checker_bindings{self.get_ext_filename('').split('.')[-1]}"
        
        # Copy the built extension to the target directory
        self.copy_file(os.path.join(ext.sourcedir, ext_file), ext_path)

setup(
    name='checker',
    version='0.1',
    author='Tensara',
    author_email='example@example.com',
    description='PyTorch to CUDA checker',
    long_description='',
    ext_modules=[MakeExtension('checker.checker_bindings')],
    cmdclass={'build_ext': MakeBuild},
    zip_safe=False,
    python_requires='>=3.6',
    install_requires=[
        'torch',
        'numpy',
        'pybind11',
    ],
) 