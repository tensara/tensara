#!/usr/bin/env python3
"""
Diagnostic test for dstack log capture
Tests if benchmark output is properly streamed and captured
"""

import sys
from pathlib import Path

# Add engine directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "engine"))

from dstack_runner import DStackClient, TaskConfig

def main():
    print("=== Dstack Log Capture Diagnostic ===")
    print()
    
    # Simple HIP program that prints benchmark-style output
    test_code = """
#include <hip/hip_runtime.h>
#include <iostream>

int main() {
    std::cout << "=== Test Benchmark Output ===" << std::endl;
    std::cout << "Runtime: 0.123 ms" << std::endl;
    std::cout << "GFLOPS: 17.5" << std::endl;
    std::cout << "Test completed successfully" << std::endl;
    return 0;
}
"""
    
    print("Creating test task configuration...")
    config = TaskConfig(
        gpu_type="MI300X",
        source_code=test_code,
        problem_id="diagnostic-test",
        submission_id="test-log-capture",
        timeout=300,  # 5 minutes
        problem_def=None,
        dtype="float32"
    )
    
    print(f"Submitting test task to dstack...")
    print()
    
    try:
        client = DStackClient()
        
        # Submit without waiting (we'll poll manually)
        result = client.submit_task(config, wait=False)
        task_id = result.task_id
        
        print(f"✓ Task submitted: {task_id}")
        print("  Waiting for provisioning and execution...")
        print()
        
        # Now wait for completion
        import time
        max_wait = 300  # 5 minutes
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            result = client.get_task_status(task_id)
            status = result.status.value
            
            print(f"  Status: {status}", end="\r")
            
            if result.status.name in ('SUCCEEDED', 'FAILED', 'TERMINATED'):
                print(f"\n✓ Task completed with status: {status}")
                print()
                break
            
            time.sleep(5)
        
        # Check output
        print("=== Output Capture Results ===")
        output = result.output or ""
        print(f"Output length: {len(output)} characters")
        print()
        
        if len(output) > 0:
            print("✓ OUTPUT CAPTURED SUCCESSFULLY!")
            print()
            print("Output content:")
            print("-" * 60)
            print(output)
            print("-" * 60)
            
            # Check for expected strings
            if "Runtime:" in output and "GFLOPS:" in output:
                print()
                print("✓✓✓ SUCCESS: Benchmark output format detected!")
                print("    The harness output SHOULD be captured correctly.")
                return 0
            else:
                print()
                print("⚠ WARNING: Output captured but missing benchmark format")
                return 1
        else:
            print("✗ FAILURE: No output captured (0 characters)")
            print()
            print("This confirms the issue: dstack is NOT capturing stdout")
            print("after task completion. We need real-time streaming.")
            return 1
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
