#!/usr/bin/env python3
"""
AMD Fleet Test Script

This script tests the AMD DevCloud fleet functionality to ensure:
1. Fleet management works correctly
2. CLI wrapper integration functions
3. End-to-end task submission works with fleets

Usage:
    python3 test_amd_fleet.py [--create-fleet] [--test-task] [--verbose]
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path

# Add engine directory to path for imports
engine_path = Path(__file__).parent / "engine"
sys.path.insert(0, str(engine_path))

# Configure logging
def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def test_fleet_manager_import():
    """Test if fleet manager can be imported and initialized"""
    print("🧪 Testing fleet manager import...")
    
    try:
        from amd_fleet_manager import AMDFleetManager, ensure_amd_fleet_ready
        print("✅ Fleet manager imported successfully")
        
        # Test initialization
        manager = AMDFleetManager()
        print(f"✅ Fleet manager initialized")
        print(f"   Fleet name: {manager.amd_config['fleet_name']}")
        print(f"   SSH user: {manager.amd_config['ssh_user']}")
        print(f"   SSH key: {manager.amd_config['ssh_key_path']}")
        print(f"   Hosts: {len(manager.amd_config['hosts'])} configured")
        
        return True
        
    except Exception as e:
        print(f"❌ Fleet manager import failed: {e}")
        return False

def test_cli_wrapper():
    """Test CLI wrapper integration"""
    print("\n🧪 Testing CLI wrapper integration...")
    
    try:
        from dstack_cli_wrapper import DStackCLIWrapper
        print("✅ CLI wrapper imported successfully")
        
        # Test initialization
        wrapper = DStackCLIWrapper()
        print("✅ CLI wrapper initialized")
        
        # Test that fleet manager is available in wrapper
        from dstack_cli_wrapper import FLEET_MANAGER_AVAILABLE
        if FLEET_MANAGER_AVAILABLE:
            print("✅ Fleet manager available in CLI wrapper")
        else:
            print("⚠️  Fleet manager not available in CLI wrapper")
        
        return True
        
    except Exception as e:
        print(f"❌ CLI wrapper test failed: {e}")
        return False

def test_fleet_status():
    """Test fleet status checking"""
    print("\n🧪 Testing fleet status...")
    
    try:
        from amd_fleet_manager import AMDFleetManager
        manager = AMDFleetManager()
        
        fleet_name = manager.amd_config['fleet_name']
        status = manager.get_fleet_status(fleet_name)
        
        print(f"✅ Fleet status retrieved for '{fleet_name}':")
        print(f"   Exists: {status.get('exists', False)}")
        print(f"   Accessible: {status.get('accessible', False)}")
        print(f"   Last checked: {status.get('last_checked', 'unknown')}")
        
        if status.get('error'):
            print(f"   Error: {status['error']}")
        
        return status.get('accessible', False)
        
    except Exception as e:
        print(f"❌ Fleet status test failed: {e}")
        return False

def test_fleet_creation():
    """Test fleet creation"""
    print("\n🧪 Testing fleet creation...")
    
    try:
        from amd_fleet_manager import ensure_amd_fleet_ready
        
        print("🔄 Attempting to ensure AMD fleet is ready...")
        success = ensure_amd_fleet_ready()
        
        if success:
            print("✅ Fleet creation/validation successful")
        else:
            print("❌ Fleet creation/validation failed")
        
        return success
        
    except Exception as e:
        print(f"❌ Fleet creation test failed: {e}")
        return False

def test_amd_task_runner():
    """Test AMD task runner with fleet integration"""
    print("\n🧪 Testing AMD task runner integration...")
    
    try:
        from amd_task_runner import execute_task
        print("✅ AMD task runner imported successfully")
        
        # Create test payload (minimal)
        test_payload = {
            'solution_code': '''
// Simple HIP test kernel
#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void test_kernel(float* data, int n) {
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}

int main() {
    printf("Test kernel executed successfully\\n");
    return 0;
}
''',
            'problem': 'test-fleet',
            'problem_def': '{"name": "fleet_test", "description": "Test fleet functionality"}',
            'gpu_type': 'MI300X',
            'dtype': 'float32',
            'endpoint': 'sandbox',  # Use sandbox for testing
            'language': 'hip',
            'submission_id': f'fleet-test-{int(time.time())}'
        }
        
        print("✅ Test payload created")
        print("⚠️  Note: Actual task execution requires valid AMD DevCloud access")
        print("   To test task execution, run with --test-task flag when AMD access is configured")
        
        return True
        
    except Exception as e:
        print(f"❌ AMD task runner test failed: {e}")
        return False

def run_full_task_test():
    """Run a full end-to-end task test (requires AMD DevCloud access)"""
    print("\n🚀 Running full end-to-end task test...")
    print("⚠️  This requires valid AMD DevCloud configuration")
    
    try:
        from amd_task_runner import execute_task
        
        # Create minimal test payload
        test_payload = {
            'solution_code': '''
#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void simple_kernel() {
    printf("Hello from MI300X GPU!\\n");
}

int main() {
    printf("=== Fleet Test Kernel ===\\n");
    
    simple_kernel<<<1, 1>>>();
    hipDeviceSynchronize();
    
    printf("✅ Fleet test completed successfully\\n");
    return 0;
}
''',
            'problem': 'fleet-integration-test',
            'problem_def': '{"name": "Fleet Integration Test", "description": "Test fleet functionality"}',
            'gpu_type': 'MI300X',
            'dtype': 'float32',
            'endpoint': 'sandbox',
            'language': 'hip',
            'submission_id': f'fleet-integration-test-{int(time.time())}'
        }
        
        print(f"🔄 Submitting fleet test task: {test_payload['submission_id']}")
        
        # Execute task
        result = execute_task(test_payload)
        
        if result == 0:
            print("✅ Full task test completed successfully!")
        else:
            print("❌ Full task test failed")
        
        return result == 0
        
    except Exception as e:
        print(f"❌ Full task test failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test AMD Fleet functionality")
    parser.add_argument('--create-fleet', action='store_true',
                       help='Test fleet creation (requires AMD DevCloud access)')
    parser.add_argument('--test-task', action='store_true',
                       help='Run full end-to-end task test (requires AMD DevCloud access)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    logger = setup_logging(args.verbose)
    
    print("🚀 AMD Fleet Test Suite")
    print("=" * 50)
    
    # Track test results
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Fleet manager import
    total_tests += 1
    if test_fleet_manager_import():
        tests_passed += 1
    
    # Test 2: CLI wrapper integration
    total_tests += 1
    if test_cli_wrapper():
        tests_passed += 1
    
    # Test 3: Fleet status
    total_tests += 1
    if test_fleet_status():
        tests_passed += 1
    
    # Test 4: AMD task runner integration
    total_tests += 1
    if test_amd_task_runner():
        tests_passed += 1
    
    # Optional Test 5: Fleet creation
    if args.create_fleet:
        total_tests += 1
        if test_fleet_creation():
            tests_passed += 1
    
    # Optional Test 6: Full task test
    if args.test_task:
        total_tests += 1
        if run_full_task_test():
            tests_passed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Test Summary: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! AMD fleet integration is ready.")
        
        if not args.create_fleet and not args.test_task:
            print("\n💡 Next steps:")
            print("   1. Configure AMD DevCloud access (run setup_amd_devcloud.sh)")
            print("   2. Test fleet creation: python3 test_amd_fleet.py --create-fleet")
            print("   3. Test full execution: python3 test_amd_fleet.py --test-task")
        
        return 0
    else:
        print("❌ Some tests failed. Check configuration and dependencies.")
        return 1

if __name__ == "__main__":
    sys.exit(main())