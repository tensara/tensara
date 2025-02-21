NVCC = nvcc
NVCCFLAGS = -O3

TARGET = checker
HEADERS = core.hpp tests.hpp solution.cuh

SRCS = checker.cu reference.cu

$(TARGET): $(SRCS) $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(SRCS) -o $(TARGET)

.PHONY: clean test

test: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) 