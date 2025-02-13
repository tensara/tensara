NVCC = nvcc
NVCC_FLAGS = -O3 -Xptxas -O3 -arch=sm_75

TARGET = benchmark

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
