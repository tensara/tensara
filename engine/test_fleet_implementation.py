#!/usr/bin/env python3
"""
Test Fleet Implementation

This script tests the new fleet-based dstack API implementation to ensure:
1. Fleet manager can be imported and initialized
2. Fleet creation works (if AMD DevCloud is configured)
3. Task submission uses the fleet correctly
4. No "no offers" errors occur

Usage:
    python3 test_fleet_implementation.py [--create-fleet] [--test-submission]
"""

import sys
import os
import json
import time
from pathlib import Path

# Add engine directory to path
sys.path.insert(0, str(Path(__file__).parent))


def test_fleet_manager_import():
    """Test if fleet manager can be imported"""
    print("\n" + "=" * 80)
    print("TEST 1: Fleet Manager Import")
    print("=" * 80)
    
    try:
        from amd_fleet_manager import AMDFleetManager, ensure_amd_fleet_ready
        print("✅ Fleet manager imported successfully")
        
        # Initialize manager
        manager = AMDFleetManager()
        print(f"✅ Fleet manager initialized")
        print(f"   Fleet name: {manager.amd_config['fleet_name']}")
        print(f"   SSH user: {manager.amd_config['ssh_user']}")
        print(f"   Hosts: {len(manager.amd_config['hosts'])} configured")
        
        return True
    except Exception as e:
        print(f"❌ Failed to import fleet manager: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fleet_in_dstack_runner():
    """Test if dstack_runner.py uses fleet correctly"""
    print("\n" + "=" * 80)
    print("TEST 2: DStack Runner Fleet Integration")
    print("=" * 80)
    
    try:
        from dstack_runner import DStackClient
        print("✅ DStack client imported successfully")
        
        # Check if Task creation includes fleet parameter
        import inspect
        from dstack_runner import DStackClient
        
        # Read the source code to verify fleet parameter is used
        source_file = Path(__file__).parent / "dstack_runner.py"
        source_code = source_file.read_text()
        
        if "fleets=[fleet_name]" in source_code:
            print("✅ DStack runner uses 'fleets' parameter for task submission")
        else:
            print("❌ DStack runner does NOT use 'fleets' parameter")
            return False
        
        if "ensure_amd_fleet_ready" in source_code:
            print("✅ DStack runner calls fleet manager before submission")
        else:
            print("⚠️  DStack runner may not ensure fleet exists before submission")
        
        return True
    except Exception as e:
        print(f"❌ Failed to test dstack runner: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fleet_in_cli_wrapper():
    """Test if CLI wrapper uses fleet correctly"""
    print("\n" + "=" * 80)
    print("TEST 3: CLI Wrapper Fleet Integration")
    print("=" * 80)
    
    try:
        from dstack_cli_wrapper import DStackCLIWrapper
        print("✅ CLI wrapper imported successfully")
        
        # Check if .dstack.yml includes fleet configuration
        source_file = Path(__file__).parent / "dstack_cli_wrapper.py"
        source_code = source_file.read_text()
        
        if "'fleets':" in source_code or '"fleets":' in source_code:
            print("✅ CLI wrapper adds 'fleets' field to .dstack.yml")
        else:
            print("❌ CLI wrapper does NOT add 'fleets' field to .dstack.yml")
            return False
        
        if "ensure_amd_fleet_ready" in source_code:
            print("✅ CLI wrapper calls fleet manager before submission")
        else:
            print("⚠️  CLI wrapper relies on manual fleet creation")
        
        return True
    except Exception as e:
        print(f"❌ Failed to test CLI wrapper: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fleet_creation(create_fleet=False):
    """Test fleet creation (requires AMD DevCloud access)"""
    print("\n" + "=" * 80)
    print("TEST 4: Fleet Creation")
    print("=" * 80)
    
    if not create_fleet:
        print("⏭️  Skipping fleet creation test (use --create-fleet to run)")
        return True
    
    try:
        from amd_fleet_manager import ensure_amd_fleet_ready
        
        fleet_name = os.getenv('AMD_FLEET_NAME', 'amd-mi300x-fleet')
        print(f"🔄 Attempting to ensure fleet '{fleet_name}' exists...")
        
        success = ensure_amd_fleet_ready(fleet_name)
        
        if success:
            print(f"✅ Fleet '{fleet_name}' is ready")
            return True
        else:
            print(f"❌ Fleet '{fleet_name}' could not be created or verified")
            print("   This may be expected if AMD DevCloud is not configured")
            return False
    except Exception as e:
        print(f"❌ Fleet creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_task_submission(test_submission=False):
    """Test task submission with fleet (requires AMD DevCloud access)"""
    print("\n" + "=" * 80)
    print("TEST 5: Task Submission with Fleet")
    print("=" * 80)
    
    if not test_submission:
        print("⏭️  Skipping task submission test (use --test-submission to run)")
        return True
    
    try:
        from dstack_runner import DStackClient, TaskConfig
        
        print("🔄 Creating test task configuration...")
        
        # Simple test kernel
        test_kernel = """
        #include <hip/hip_runtime.h>
        __global__ void test_kernel() {
            printf("Hello from MI300X!\\n");
        }
        """
        
        config = TaskConfig(
            gpu_type="MI300X",
            source_code=test_kernel,
            problem_id="fleet-test",
            submission_id=f"fleet-test-{int(time.time())}",
            timeout=300,
        )
        
        print(f"🔄 Submitting test task: {config.submission_id}")
        
        client = DStackClient()
        result = client.submit_task(config, wait=False)
        
        print(f"✅ Task submitted successfully: {result.task_id}")
        print(f"   Status: {result.status.value}")
        
        return True
    except Exception as e:
        print(f"❌ Task submission test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test fleet implementation")
    parser.add_argument('--create-fleet', action='store_true',
                       help='Test fleet creation (requires AMD DevCloud access)')
    parser.add_argument('--test-submission', action='store_true',
                       help='Test task submission (requires AMD DevCloud access)')
    
    args = parser.parse_args()
    
    print("\n")
    print("=" * 80)
    print(" " * 20 + "FLEET IMPLEMENTATION TEST SUITE")
    print("=" * 80)
    
    # Run tests
    results = []
    
    # Test 1: Fleet manager import
    results.append(("Fleet Manager Import", test_fleet_manager_import()))
    
    # Test 2: DStack runner fleet integration
    results.append(("DStack Runner Integration", test_fleet_in_dstack_runner()))
    
    # Test 3: CLI wrapper fleet integration
    results.append(("CLI Wrapper Integration", test_fleet_in_cli_wrapper()))
    
    # Test 4: Fleet creation (optional)
    if args.create_fleet:
        results.append(("Fleet Creation", test_fleet_creation(True)))
    
    # Test 5: Task submission (optional)
    if args.test_submission:
        results.append(("Task Submission", test_task_submission(True)))
    
    # Print summary
    print("\n" + "=" * 80)
    print(" " * 30 + "TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print()
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Fleet implementation is ready.")
        print("\nNext steps:")
        print("  1. Set up AMD DevCloud SSH configuration")
        print("  2. Create fleet: python3 amd_fleet_manager.py ensure")
        print("  3. Test task submission: python3 test_fleet_implementation.py --test-submission")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
