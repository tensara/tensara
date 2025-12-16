"""
Client wrapper for VM Orchestrator
Provides simple interface for task submission
"""

import time
from typing import Optional
from vm_orchestrator import VMOrchestrator
from dstack_runner import TaskConfig, TaskResult

# Singleton instance
_orchestrator: Optional[VMOrchestrator] = None


def get_orchestrator() -> VMOrchestrator:
    """Get or create orchestrator instance"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = VMOrchestrator()
    return _orchestrator


def submit_hip_kernel(
    solution_code: str,
    problem: str,
    problem_def: str,
    gpu_type: str = "MI300X",
    submission_id: Optional[str] = None,
    dtype: str = "float32",
    language: str = "hip",
    endpoint: str = "checker",
) -> TaskResult:
    """
    Submit HIP kernel with VM orchestration
    
    Automatically handles:
    - Finding warm VM or provisioning new one
    - Executing in isolated Docker container
    - Updating VM state and cost tracking
    
    Args:
        solution_code: HIP kernel source code
        problem: Problem slug
        problem_def: Problem definition
        gpu_type: GPU type (default: MI300X)
        submission_id: Optional submission ID
        dtype: Data type (default: float32)
        language: Language (default: hip)
        endpoint: Endpoint type (checker, benchmark, etc.)
    
    Returns:
        TaskResult with execution results
    """
    orchestrator = get_orchestrator()
    
    # Get or create VM
    vm = orchestrator.get_or_create_vm(gpu_type)
    
    # Create task config
    task_config = TaskConfig(
        gpu_type=gpu_type,
        source_code=solution_code,
        problem_id=problem,
        submission_id=submission_id or f"task-{int(time.time())}",
        timeout=600,
    )
    
    # Submit to VM
    return orchestrator.submit_task(vm, task_config)


def get_orchestrator_metrics():
    """Get current metrics"""
    orchestrator = get_orchestrator()
    return orchestrator.get_metrics()
