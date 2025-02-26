NVCC = nvcc
NVCCFLAGS = -O3

CXX = g++
CXXFLAGS = -O3 -std=c++14 -fPIC

# Get Python include directory
PYTHON_INCLUDE = $(shell python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
PYTHON_LIB = $(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
PYTHON_LDFLAGS = -L$(PYTHON_LIB) -lpython3

# Get pybind11 include directory
PYBIND11_INCLUDE = $(shell python3 -c "import pybind11; print(pybind11.get_include())")

TARGET = checker
HEADERS = core.hpp tests.hpp python_reference.hpp
SRCS = checker.cu python_reference.cpp

# For Python module
MODULE_NAME = checker_bindings
MODULE_SRCS = bindings.cpp python_reference.cpp
MODULE_TARGET = $(MODULE_NAME)$(shell python3-config --extension-suffix)

.PHONY: all clean test python

all: $(TARGET) $(MODULE_TARGET)

$(TARGET): $(SRCS) $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(SRCS) -o $(TARGET)

$(MODULE_TARGET): $(MODULE_SRCS) $(HEADERS)
	$(CXX) $(CXXFLAGS) -shared -I$(PYTHON_INCLUDE) -I$(PYBIND11_INCLUDE) $(MODULE_SRCS) -o $(MODULE_TARGET) $(PYTHON_LDFLAGS)

test: $(TARGET)
	./$(TARGET)

python: $(MODULE_TARGET)
	python3 reference.py

clean:
	rm -f $(TARGET) $(MODULE_TARGET) 