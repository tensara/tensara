#!/bin/bash
# Diagnostic test for dstack stdout capture
# This tests if HIP compilation and stdout output work on MI300X

echo "=== Dstack Output Capture Diagnostic Test ==="
echo ""
echo "Creating simple HIP test program..."

cat > /tmp/test_output.hip << 'EOF'
#include <hip/hip_runtime.h>
#include <iostream>

int main() {
    std::cout << "=== Test Output Capture ===" << std::endl;
    std::cout << "HIP program executed successfully" << std::endl;
    std::cout << "Runtime: 0.123 ms" << std::endl;
    std::cout << "GFLOPS: 17.5" << std::endl;
    std::cout << "=== End Test ===" << std::endl;
    return 0;
}
EOF

echo "Submitting to dstack..."
echo ""

# Submit task and capture output
dstack run \
  --gpu MI300X \
  --image rocm/pytorch:latest \
  -- bash -c '
    echo "=== ROCm Environment ==="
    hipcc --version || echo "hipcc not available"
    echo ""
    echo "=== Compiling Test Program ==="
    cat > test.hip << '"'"'EOFHIP'"'"'
#include <hip/hip_runtime.h>
#include <iostream>

int main() {
    std::cout << "=== Test Output Capture ===" << std::endl;
    std::cout << "HIP program executed successfully" << std::endl;
    std::cout << "Runtime: 0.123 ms" << std::endl;
    std::cout << "GFLOPS: 17.5" << std::endl;
    std::cout << "=== End Test ===" << std::endl;
    return 0;
}
EOFHIP
    
    hipcc test.hip -o test -O3 || exit 1
    echo "Compilation successful"
    echo ""
    echo "=== Running Test Program ==="
    ./test
    echo ""
    echo "=== Test Complete ==="
  '

echo ""
echo "=== Diagnostic Test Complete ==="
echo "If you see 'Runtime: 0.123 ms' and 'GFLOPS: 17.5' above, stdout capture works!"
