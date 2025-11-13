"""
dstack.ai Official SDK Wrapper for Tensara AMD GPU Orchestration

This module provides a high-level interface using the official dstack Python SDK for:
- Submitting ROCm/HIP kernel execution tasks
- Monitoring task status and progress
- Retrieving execution results
- Managing VM lifecycle
- Tracking costs and resource usage

Based on Phase 1 Section 1.1 of the AMD/ROCm Implementation Plan.
Uses the official dstack SDK from PyPI.
"""

import os
import json
import time
import logging
import hashlib
import sys
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum

# Official dstack SDK imports
try:
    from dstack.api import Client, Task, Resources, GPU
except ImportError:
    raise ImportError(
        "dstack SDK not installed. Install with: pip install dstack"
    )


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    PROVISIONING = "provisioning"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TERMINATED = "terminated"
    TIMEOUT = "timeout"


class GPUType(Enum):
    """Supported AMD GPU types"""
    MI210 = "MI210"
    MI250X = "MI250X"
    MI300A = "MI300A"
    MI300X = "MI300X"


@dataclass
class TaskConfig:
    """Configuration for a dstack.ai task"""
    gpu_type: str
    source_code: str
    problem_id: str
    submission_id: str
    timeout: int = 600  # 10 minutes default
    profile: str = "mi210-standard"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def source_hash(self) -> str:
        """Generate hash of source code for caching"""
        return hashlib.sha256(self.source_code.encode()).hexdigest()[:16]


@dataclass
class TaskResult:
    """Result of a task execution"""
    task_id: str
    status: TaskStatus
    output: Optional[str] = None
    error: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    cost_usd: Optional[float] = None
    execution_time: Optional[float] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        result['status'] = self.status.value
        if self.created_at:
            result['created_at'] = self.created_at.isoformat()
        if self.completed_at:
            result['completed_at'] = self.completed_at.isoformat()
        return result


class DStackError(Exception):
    """Base exception for dstack.ai errors"""
    pass


class ProvisioningError(DStackError):
    """Error during VM provisioning"""
    pass


class ExecutionError(DStackError):
    """Error during task execution"""
    pass


class TimeoutError(DStackError):
    """Task execution timeout"""
    pass


class DStackClient:
    """
    High-level client for interacting with dstack.ai using official SDK
    
    Handles authentication, task submission, monitoring, and cost tracking.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        workspace: Optional[str] = None,
        timeout: int = 30,
    ):
        """
        Initialize dstack client using official SDK
        
        Args:
            api_key: Not used - SDK uses config files
            base_url: Not used - SDK uses config files
            workspace: Not used - SDK uses config files
            timeout: HTTP request timeout in seconds
        
        Note: The official SDK reads configuration from:
            - ~/.dstack/config.yml (or DSTACK_CONFIG_PATH)
            - Environment variables (DSTACK_TOKEN, etc.)
        """
        # Initialize official SDK client
        try:
            self.client = Client.from_config()
            logger.info("DStack SDK client initialized from config")
        except Exception as e:
            logger.error(f"Failed to initialize dstack client: {e}")
            raise DStackError(
                f"Failed to initialize dstack SDK: {e}\n"
                "Make sure dstack is configured. Run: dstack config"
            ) from e
        
        # Cost tracking
        self.total_cost = 0.0
        self.task_costs: Dict[str, float] = {}
        
        # Store run references for cleanup
        self._active_runs: Dict[str, Any] = {}
    
    def submit_task(
        self,
        config: TaskConfig,
        wait: bool = False,
        poll_interval: int = 5,
    ) -> TaskResult:
        """
        Submit a ROCm kernel execution task using official SDK
        
        Args:
            config: Task configuration
            wait: If True, wait for task completion
            poll_interval: Polling interval in seconds when waiting
            
        Returns:
            TaskResult with task ID and initial status
            
        Raises:
            DStackError: If submission fails
        """
        logger.info(f"Submitting task for {config.problem_id}/{config.submission_id}")
        
        try:
            # Prepare task using official SDK
            # Create commands to:
            # 1. Write the HIP source code to file
            # 2. Compile with hipcc
            # 3. Run the kernel
            commands = [
                # Write source code to file
                f"cat > solution.hip << 'EOFHIP'\n{config.source_code}\nEOFHIP",
                # Check ROCm installation
                "echo '=== ROCm Environment ==='",
                "rocm-smi --showproductname || echo 'rocm-smi not available'",
                "hipcc --version || echo 'hipcc not available'",
                "echo",
                # Compile the kernel
                "echo '=== Compiling HIP Kernel ==='",
                "hipcc solution.hip -o solution -O3 || exit 1",
                "echo 'Compilation successful'",
                "echo",
                # Run the kernel
                "echo '=== Running Kernel ==='",
                "./solution 1024",  # Default matrix size
                "echo",
                "echo '=== Execution Complete ==='",
            ]
            
            # Map GPU types to memory requirements
            gpu_memory_map = {
                "MI210": "64GB",
                "MI250X": "128GB",
                "MI300A": "192GB",
                "MI300X": "192GB",
            }
            gpu_memory = gpu_memory_map.get(config.gpu_type, "64GB")
            
            # Create task using official SDK
            task = Task(
                name=f"tensara-{config.submission_id}",
                image="rocm/pytorch:latest",  # ROCm base image with HIP support
                env={
                    "GPU_TYPE": config.gpu_type,
                    "SOURCE_HASH": config.source_hash(),
                    "PROBLEM_ID": config.problem_id,
                    "SUBMISSION_ID": config.submission_id,
                    "ROCM_PATH": "/opt/rocm",
                    "HIP_PLATFORM": "amd",
                },
                commands=commands,
                resources=Resources(
                    gpu=GPU(memory=gpu_memory, name=config.gpu_type),
                ),
            )
            
            # Submit the task
            logger.info(f"Applying configuration for {task.name}")
            run = self.client.runs.submit(
                configuration=task,
                repo=None,  # No repo needed for inline tasks
            )
            
            # Store run reference
            task_id = run.name
            self._active_runs[task_id] = run
            
            logger.info(f"Task submitted successfully: {task_id}")
            
            result = TaskResult(
                task_id=task_id,
                status=TaskStatus.PENDING,
                created_at=datetime.utcnow(),
            )
            
            # Wait for completion if requested
            if wait:
                result = self.wait_for_completion(task_id, poll_interval, config.timeout)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to submit task: {e}")
            raise DStackError(f"Task submission failed: {e}") from e
    
    def get_task_status(self, task_id: str) -> TaskResult:
        """
        Get current status of a task
        
        Args:
            task_id: Task identifier (run name)
            
        Returns:
            TaskResult with current status and available data
            
        Raises:
            DStackError: If status check fails
        """
        try:
            # Get run from stored reference or list runs
            run = self._active_runs.get(task_id)
            
            if not run:
                # Try to find the run by listing
                runs = self.client.runs.list()
                for r in runs:
                    if r.name == task_id:
                        run = r
                        self._active_runs[task_id] = run
                        break
            
            if not run:
                raise DStackError(f"Task {task_id} not found")
            
            # Map dstack run status to our TaskStatus
            status_map = {
                "submitted": TaskStatus.PENDING,
                "pending": TaskStatus.PENDING,
                "provisioning": TaskStatus.PROVISIONING,
                "running": TaskStatus.RUNNING,
                "done": TaskStatus.SUCCEEDED,
                "failed": TaskStatus.FAILED,
                "terminated": TaskStatus.TERMINATED,
                "terminating": TaskStatus.TERMINATED,
            }
            
            run_status = getattr(run, 'status', 'pending').lower()
            status = status_map.get(run_status, TaskStatus.PENDING)
            
            # Get execution details
            created_at = getattr(run, 'submitted_at', None)
            completed_at = getattr(run, 'finished_at', None)
            
            # Calculate execution time
            execution_time = None
            if created_at and completed_at:
                try:
                    execution_time = (completed_at - created_at).total_seconds()
                except:
                    pass
            
            # Calculate cost based on GPU time
            cost_usd = None
            if execution_time:
                # Try to extract GPU type from run environment
                gpu_type = "MI210"  # Default
                try:
                    env = getattr(run, 'env', {})
                    gpu_type = env.get('GPU_TYPE', 'MI210')
                except:
                    pass
                
                cost_usd = self._calculate_cost_from_time(gpu_type, execution_time)
                if cost_usd:
                    self.task_costs[task_id] = cost_usd
                    self.total_cost += cost_usd
            
            # Get output/logs if available
            output = None
            error = None
            
            if status in (TaskStatus.SUCCEEDED, TaskStatus.FAILED):
                try:
                    output = self.get_task_logs(task_id)
                except:
                    pass
                
                if status == TaskStatus.FAILED:
                    error = getattr(run, 'error_message', 'Task failed')
            
            return TaskResult(
                task_id=task_id,
                status=status,
                output=output,
                error=error,
                metrics=None,
                cost_usd=cost_usd,
                execution_time=execution_time,
                created_at=created_at,
                completed_at=completed_at,
            )
            
        except Exception as e:
            logger.error(f"Failed to get task status: {e}")
            raise DStackError(f"Status check failed: {e}") from e
    
    def wait_for_completion(
        self,
        task_id: str,
        poll_interval: int = 5,
        timeout: int = 600,
    ) -> TaskResult:
        """
        Wait for task to complete
        
        Args:
            task_id: Task identifier
            poll_interval: Polling interval in seconds
            timeout: Maximum wait time in seconds
            
        Returns:
            TaskResult with final status
            
        Raises:
            TimeoutError: If task doesn't complete within timeout
            DStackError: If task fails
        """
        start_time = time.time()
        logger.info(f"Waiting for task {task_id} to complete (timeout: {timeout}s)")
        
        run = self._active_runs.get(task_id)
        if run:
            try:
                # Use SDK's attach to follow execution
                run.attach()
                
                # Stream logs
                logger.info("Streaming logs...")
                for log_chunk in run.logs():
                    # Logs are bytes, decode and print
                    try:
                        sys.stdout.buffer.write(log_chunk)
                        sys.stdout.buffer.flush()
                    except:
                        # Fallback to text output
                        print(log_chunk.decode('utf-8', errors='ignore'), end='')
                
            except Exception as e:
                logger.warning(f"Log streaming failed: {e}")
        
        # Poll for completion
        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                try:
                    self.terminate_task(task_id)
                except:
                    pass
                raise TimeoutError(f"Task {task_id} timed out after {timeout}s")
            
            result = self.get_task_status(task_id)
            
            if result.status in (TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.TERMINATED):
                logger.info(f"Task {task_id} completed with status: {result.status.value}")
                return result
            
            logger.debug(f"Task {task_id} status: {result.status.value} (elapsed: {elapsed:.1f}s)")
            time.sleep(poll_interval)
    
    def get_task_logs(self, task_id: str) -> str:
        """
        Retrieve task execution logs
        
        Args:
            task_id: Task identifier
            
        Returns:
            Log output as string
            
        Raises:
            DStackError: If log retrieval fails
        """
        try:
            run = self._active_runs.get(task_id)
            if not run:
                runs = self.client.runs.list()
                for r in runs:
                    if r.name == task_id:
                        run = r
                        break
            
            if not run:
                raise DStackError(f"Task {task_id} not found")
            
            # Collect all logs
            log_output = []
            for log_chunk in run.logs():
                try:
                    log_output.append(log_chunk.decode('utf-8', errors='ignore'))
                except:
                    log_output.append(str(log_chunk))
            
            return ''.join(log_output)
            
        except Exception as e:
            logger.error(f"Failed to get task logs: {e}")
            raise DStackError(f"Log retrieval failed: {e}") from e
    
    def terminate_task(self, task_id: str) -> bool:
        """
        Terminate a running task
        
        Args:
            task_id: Task identifier
            
        Returns:
            True if termination successful
            
        Raises:
            DStackError: If termination fails
        """
        logger.info(f"Terminating task {task_id}")
        
        try:
            run = self._active_runs.get(task_id)
            if not run:
                runs = self.client.runs.list()
                for r in runs:
                    if r.name == task_id:
                        run = r
                        break
            
            if not run:
                logger.warning(f"Task {task_id} not found for termination")
                return False
            
            # Terminate the run
            run.stop()
            logger.info(f"Task {task_id} terminated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to terminate task: {e}")
            raise DStackError(f"Termination failed: {e}") from e
    
    def list_tasks(
        self,
        limit: int = 10,
        status: Optional[TaskStatus] = None,
    ) -> List[TaskResult]:
        """
        List recent tasks
        
        Args:
            limit: Maximum number of tasks to return
            status: Filter by status (optional)
            
        Returns:
            List of TaskResult objects
            
        Raises:
            DStackError: If listing fails
        """
        try:
            runs = self.client.runs.list()
            
            results = []
            for run in runs[:limit]:
                # Map status
                status_map = {
                    "submitted": TaskStatus.PENDING,
                    "pending": TaskStatus.PENDING,
                    "provisioning": TaskStatus.PROVISIONING,
                    "running": TaskStatus.RUNNING,
                    "done": TaskStatus.SUCCEEDED,
                    "failed": TaskStatus.FAILED,
                    "terminated": TaskStatus.TERMINATED,
                }
                
                run_status = getattr(run, 'status', 'pending').lower()
                task_status = status_map.get(run_status, TaskStatus.PENDING)
                
                # Filter by status if requested
                if status and task_status != status:
                    continue
                
                results.append(TaskResult(
                    task_id=run.name,
                    status=task_status,
                    output=None,
                    error=None,
                    cost_usd=None,
                    execution_time=None,
                ))
                
                if len(results) >= limit:
                    break
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to list tasks: {e}")
            raise DStackError(f"Task listing failed: {e}") from e
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """
        Get cost summary for all tasks
        
        Returns:
            Dictionary with cost statistics
        """
        return {
            "total_cost_usd": round(self.total_cost, 4),
            "task_count": len(self.task_costs),
            "average_cost_usd": round(
                self.total_cost / len(self.task_costs) if self.task_costs else 0.0,
                4
            ),
            "task_costs": {
                task_id: round(cost, 4)
                for task_id, cost in self.task_costs.items()
            },
        }
    
    def _calculate_cost_from_time(self, gpu_type: str, execution_time: float) -> Optional[float]:
        """
        Calculate task cost based on GPU type and execution time
        
        Args:
            gpu_type: GPU type string
            execution_time: Execution time in seconds
            
        Returns:
            Cost in USD, or None if cannot be calculated
        """
        # Cost rates per hour (from implementation plan)
        COST_RATES = {
            "MI210": 3.00,
            "MI250X": 5.00,
            "MI300A": 7.00,
            "MI300X": 8.00,
        }
        
        if not execution_time:
            return None
        
        hourly_rate = COST_RATES.get(gpu_type, 3.00)  # Default to MI210 rate
        cost = (execution_time / 3600.0) * hourly_rate
        
        return round(cost, 4)
    
    def _calculate_cost(self, task_data: Dict[str, Any]) -> Optional[float]:
        """
        Calculate task cost based on GPU type and execution time
        
        Args:
            task_data: Task data from API response
            
        Returns:
            Cost in USD, or None if cannot be calculated
        """
        gpu_type = task_data.get("env", {}).get("GPU_TYPE")
        execution_time = task_data.get("execution_time")  # in seconds
        
        if not execution_time:
            return None
        
        return self._calculate_cost_from_time(gpu_type or "MI210", execution_time)
    
    def health_check(self) -> bool:
        """
        Check if dstack.ai API is accessible
        
        Returns:
            True if API is healthy
        """
        try:
            # Try to list runs as a health check
            self.client.runs.list()
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def close(self):
        """Close client and cleanup"""
        self._active_runs.clear()
        logger.info("DStack client closed")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


class TaskBatcher:
    """
    Batch multiple submissions for cost optimization
    
    Groups multiple kernel submissions to run on the same VM,
    reducing provisioning overhead.
    """
    
    def __init__(
        self,
        client: DStackClient,
        batch_size: int = 5,
        wait_time: int = 30,
    ):
        """
        Initialize task batcher
        
        Args:
            client: DStack client instance
            batch_size: Maximum batch size
            wait_time: Maximum wait time for batch to fill (seconds)
        """
        self.client = client
        self.batch_size = batch_size
        self.wait_time = wait_time
        self.pending_tasks: List[TaskConfig] = []
        self.batch_start_time: Optional[float] = None
    
    def add_task(self, config: TaskConfig) -> Optional[List[TaskResult]]:
        """
        Add task to batch
        
        Args:
            config: Task configuration
            
        Returns:
            List of TaskResults if batch is ready, None otherwise
        """
        if not self.pending_tasks:
            self.batch_start_time = time.time()
        
        self.pending_tasks.append(config)
        
        # Check if batch is ready
        if self._is_batch_ready():
            return self.process_batch()
        
        return None
    
    def _is_batch_ready(self) -> bool:
        """Check if batch is ready to process"""
        if len(self.pending_tasks) >= self.batch_size:
            return True
        
        if self.batch_start_time:
            elapsed = time.time() - self.batch_start_time
            if elapsed >= self.wait_time:
                return True
        
        return False
    
    def process_batch(self) -> List[TaskResult]:
        """
        Process batch of tasks
        
        Returns:
            List of TaskResults
        """
        if not self.pending_tasks:
            return []
        
        logger.info(f"Processing batch of {len(self.pending_tasks)} tasks")
        
        results = []
        for config in self.pending_tasks:
            try:
                result = self.client.submit_task(config, wait=True)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch task failed: {e}")
                results.append(TaskResult(
                    task_id="",
                    status=TaskStatus.FAILED,
                    error=str(e),
                ))
        
        # Clear batch
        self.pending_tasks.clear()
        self.batch_start_time = None
        
        return results
    
    def flush(self) -> List[TaskResult]:
        """Force process current batch"""
        return self.process_batch()


# Convenience functions

def submit_rocm_kernel(
    gpu_type: str,
    source_code: str,
    problem_id: str,
    submission_id: str,
    timeout: int = 600,
    wait: bool = True,
) -> TaskResult:
    """
    Convenience function to submit a ROCm kernel
    
    Args:
        gpu_type: AMD GPU type (MI210, MI250X, etc.)
        source_code: HIP kernel source code
        problem_id: Problem identifier
        submission_id: Submission identifier
        timeout: Execution timeout in seconds
        wait: Wait for completion
        
    Returns:
        TaskResult
    """
    config = TaskConfig(
        gpu_type=gpu_type,
        source_code=source_code,
        problem_id=problem_id,
        submission_id=submission_id,
        timeout=timeout,
        profile=f"{gpu_type.lower()}-standard",
    )
    
    with DStackClient() as client:
        return client.submit_task(config, wait=wait)


def get_task_result(task_id: str) -> TaskResult:
    """
    Convenience function to get task result
    
    Args:
        task_id: Task identifier
        
    Returns:
        TaskResult
    """
    with DStackClient() as client:
        return client.get_task_status(task_id)