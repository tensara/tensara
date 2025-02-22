NVCC = nvcc
NVCCFLAGS = -O3

TARGET = checker
HEADERS = core.hpp tests.hpp

SRCS = checker.cu

$(TARGET): $(SRCS) $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(SRCS) -o $(TARGET)

.PHONY: clean test

test: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) 