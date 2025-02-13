#!/bin/bash

mkdir -p build
cd build

if [ ! -d "pybind11" ]; then
    git clone https://github.com/pybind/pybind11.git
fi

cmake ..
make -j$(nproc)

cp cuda_solution*.so ..

cd .. 