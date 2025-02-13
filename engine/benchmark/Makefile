NVCC = nvcc
NVCC_FLAGS = -arch=sm_75 

# Output binary name
TARGET = benchmark

# Source files
SRC = benchmark.cu
HEADERS = solution.cuh

$(TARGET): $(SRC) $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(SRC) -o $(TARGET)

.PHONY: clean run

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

all: $(TARGET)
