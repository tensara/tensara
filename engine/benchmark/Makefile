NVCC = nvcc
NVCC_FLAGS = -O3 -Xptxas -O3 -arch=sm_75

TARGET = benchmark

HEADERS = core.hpp tests.hpp solution.cuh

$(TARGET): benchmark.cu $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) $< -o $@

.PHONY: clean run all

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

all: $(TARGET)
