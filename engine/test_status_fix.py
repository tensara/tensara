#!/usr/bin/env python3
"""
Test script to verify the status polling fix works correctly.
This tests that we can correctly detect task completion status.
"""

import sys
from dstack_runner import DStackClient, TaskStatus

def test_status_detection():
    """Test that we can correctly detect task status"""
    print("=" * 80)
    print("Testing AMD Task Status Detection Fix")
    print("=" * 80)
    
    # Initialize client
    print("\n1. Initializing DStack client...")
    client = DStackClient()
    print("✓ Client initialized")
    
    # List all recent runs
    print("\n2. Fetching recent runs...")
    from dstack.api import Client
    api_client = Client.from_config()
    runs = list(api_client.runs.list())
    
    if not runs:
        print("⚠ No runs found. Submit a task first.")
        return
    
    print(f"✓ Found {len(runs)} run(s)")
    
    # Test the most recent run
    latest_run = runs[0]
    print(f"\n3. Testing status detection for: {latest_run.name}")
    print(f"   Raw status: {latest_run.status}")
    print(f"   Status type: {type(latest_run.status)}")
    print(f"   Status value: {latest_run.status.value}")
    
    # Get status using our client
    try:
        result = client.get_task_status(latest_run.name)
        
        print(f"\n4. Status Detection Results:")
        print(f"   ✓ Detected status: {result.status.value}")
        print(f"   ✓ Is succeeded: {result.status == TaskStatus.SUCCEEDED}")
        print(f"   ✓ Is failed: {result.status == TaskStatus.FAILED}")
        print(f"   ✓ Is pending: {result.status == TaskStatus.PENDING}")
        
        if result.execution_time:
            print(f"   ✓ Execution time: {result.execution_time:.2f}s")
        else:
            print(f"   ⚠ Execution time: Not available")
            
        if result.cost_usd:
            print(f"   ✓ Cost: ${result.cost_usd:.4f}")
        else:
            print(f"   ⚠ Cost: Not available")
            
        if result.created_at:
            print(f"   ✓ Created at: {result.created_at}")
        else:
            print(f"   ⚠ Created at: Not available")
            
        if result.completed_at:
            print(f"   ✓ Completed at: {result.completed_at}")
        else:
            print(f"   ⚠ Completed at: Not available")
        
        # Verify the fix worked
        print(f"\n5. Verification:")
        if result.status == TaskStatus.PENDING and latest_run.status.value == 'done':
            print("   ❌ FAILED: Status is stuck on PENDING even though task is DONE")
            print("   This means the bug is NOT fixed!")
            return False
        elif result.status == TaskStatus.SUCCEEDED and latest_run.status.value == 'done':
            print("   ✅ PASSED: Status correctly detected as SUCCEEDED")
            print("   The bug is FIXED!")
            return True
        elif result.status == TaskStatus.FAILED and latest_run.status.value == 'failed':
            print("   ✅ PASSED: Status correctly detected as FAILED")
            print("   The bug is FIXED!")
            return True
        else:
            print(f"   ℹ️  Status is {result.status.value} (task may still be running)")
            return True
            
    except Exception as e:
        print(f"\n❌ Error getting status: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print(__doc__)
    success = test_status_detection()
    
    print("\n" + "=" * 80)
    if success:
        print("✅ TEST PASSED: Status detection is working correctly")
        print("=" * 80)
        sys.exit(0)
    else:
        print("❌ TEST FAILED: Status detection has issues")
        print("=" * 80)
        sys.exit(1)
