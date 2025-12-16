#!/usr/bin/env python3
"""
Test script to verify that 'terminated' status with exit code 0 is correctly
detected as SUCCESS, not ERROR.

This simulates what happens during the polling loop when a task goes through
the terminating phase before reaching the final 'done' state.
"""

import sys
from dstack_runner import DStackClient, TaskStatus
from unittest.mock import Mock, patch

def test_terminated_with_success():
    """Test that terminated status with exit code 0 is detected as SUCCESS"""
    print("=" * 80)
    print("Testing 'terminated' status with exit code 0 (should be SUCCESS)")
    print("=" * 80)
    
    client = DStackClient()
    
    # Create a mock run object that looks like it's terminated but actually succeeded
    mock_run = Mock()
    mock_run.name = "test-task-123"
    
    # Public status shows 'terminated'
    mock_run.status = Mock()
    mock_run.status.value = 'terminated'
    
    # But internal _run object shows exit_status = 0 (success!)
    mock_internal_run = Mock()
    mock_internal_run.submitted_at = None
    mock_internal_run.termination_reason = 'all_jobs_done'
    mock_internal_run.cost = 0.25
    
    mock_job = Mock()
    mock_job.finished_at = None
    mock_job.exit_status = 0  # ← This is the key: exit code 0 = success
    mock_job.status = Mock()
    mock_job.status.value = 'done'
    
    mock_internal_run.latest_job_submission = mock_job
    mock_run._run = mock_internal_run
    
    # Mock the client.runs.list() to return our mock run
    with patch.object(client.client.runs, 'list', return_value=[mock_run]):
        result = client.get_task_status("test-task-123")
        
        print(f"\nMock run status: terminated")
        print(f"Mock exit_status: 0")
        print(f"Mock termination_reason: all_jobs_done")
        print(f"\nDetected TaskStatus: {result.status}")
        print(f"Is SUCCEEDED: {result.status == TaskStatus.SUCCEEDED}")
        print(f"Is TERMINATED: {result.status == TaskStatus.TERMINATED}")
        
        if result.status == TaskStatus.SUCCEEDED:
            print("\n✅ PASSED: Correctly detected as SUCCEEDED")
            return True
        else:
            print(f"\n❌ FAILED: Detected as {result.status.value} instead of SUCCEEDED")
            return False

def test_terminated_with_failure():
    """Test that terminated status with exit code 1 is detected as FAILED"""
    print("\n" + "=" * 80)
    print("Testing 'terminated' status with exit code 1 (should be FAILED)")
    print("=" * 80)
    
    client = DStackClient()
    
    # Create a mock run object that's terminated with failure
    mock_run = Mock()
    mock_run.name = "test-task-456"
    
    # Public status shows 'terminated'
    mock_run.status = Mock()
    mock_run.status.value = 'terminated'
    
    # Internal _run object shows exit_status = 1 (failure!)
    mock_internal_run = Mock()
    mock_internal_run.submitted_at = None
    mock_internal_run.termination_reason = None
    mock_internal_run.cost = 0.10
    
    mock_job = Mock()
    mock_job.finished_at = None
    mock_job.exit_status = 1  # ← exit code 1 = failure
    mock_job.status = Mock()
    mock_job.status.value = 'failed'
    
    mock_internal_run.latest_job_submission = mock_job
    mock_run._run = mock_internal_run
    
    # Mock the client.runs.list() to return our mock run
    with patch.object(client.client.runs, 'list', return_value=[mock_run]):
        result = client.get_task_status("test-task-456")
        
        print(f"\nMock run status: terminated")
        print(f"Mock exit_status: 1")
        print(f"Mock termination_reason: None")
        print(f"\nDetected TaskStatus: {result.status}")
        print(f"Is FAILED: {result.status == TaskStatus.FAILED}")
        print(f"Is TERMINATED: {result.status == TaskStatus.TERMINATED}")
        
        if result.status == TaskStatus.FAILED:
            print("\n✅ PASSED: Correctly detected as FAILED")
            return True
        else:
            print(f"\n❌ FAILED: Detected as {result.status.value} instead of FAILED")
            return False

def test_terminating_no_exit_code():
    """Test that terminating status without exit code yet is treated as RUNNING"""
    print("\n" + "=" * 80)
    print("Testing 'terminating' status without exit code (should keep polling)")
    print("=" * 80)
    
    client = DStackClient()
    
    # Create a mock run object that's still terminating
    mock_run = Mock()
    mock_run.name = "test-task-789"
    
    # Public status shows 'terminating'
    mock_run.status = Mock()
    mock_run.status.value = 'terminating'
    
    # Internal _run object has no exit_status yet (still cleaning up)
    mock_internal_run = Mock()
    mock_internal_run.submitted_at = None
    mock_internal_run.termination_reason = None
    mock_internal_run.cost = None
    
    mock_job = Mock()
    mock_job.finished_at = None
    mock_job.exit_status = None  # ← No exit code yet, still terminating
    mock_job.status = Mock()
    mock_job.status.value = 'terminating'
    
    mock_internal_run.latest_job_submission = mock_job
    mock_run._run = mock_internal_run
    
    # Mock the client.runs.list() to return our mock run
    with patch.object(client.client.runs, 'list', return_value=[mock_run]):
        result = client.get_task_status("test-task-789")
        
        print(f"\nMock run status: terminating")
        print(f"Mock exit_status: None")
        print(f"Mock termination_reason: None")
        print(f"\nDetected TaskStatus: {result.status}")
        print(f"Is RUNNING: {result.status == TaskStatus.RUNNING}")
        
        if result.status == TaskStatus.RUNNING:
            print("\n✅ PASSED: Correctly treated as RUNNING (will keep polling)")
            return True
        else:
            print(f"\n❌ FAILED: Detected as {result.status.value} instead of RUNNING")
            return False

if __name__ == '__main__':
    print(__doc__)
    
    test1 = test_terminated_with_success()
    test2 = test_terminated_with_failure()
    test3 = test_terminating_no_exit_code()
    
    print("\n" + "=" * 80)
    if test1 and test2 and test3:
        print("✅ ALL TESTS PASSED: Status detection correctly handles terminated states")
        print("=" * 80)
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED: Status detection has issues")
        print("=" * 80)
        sys.exit(1)
