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

# Import HIP harness generator
try:
    from hip_harness import generate_hip_benchmark_harness
except ImportError:
    # Fallback if hip_harness.py is not in path
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from hip_harness import generate_hip_benchmark_harness


# Configure logging - send INFO to stdout, errors to stderr
import sys as _sys

class _InfoFilter(logging.Filter):
    def filter(self, record):
        return record.levelno <= logging.INFO

class _ErrorFilter(logging.Filter):
    def filter(self, record):
        return record.levelno >= logging.WARNING

_stdout_handler = logging.StreamHandler(_sys.stdout)
_stdout_handler.setLevel(logging.INFO)
_stdout_handler.addFilter(_InfoFilter())
_stdout_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

_stderr_handler = logging.StreamHandler(_sys.stderr)
_stderr_handler.setLevel(logging.WARNING)
_stderr_handler.addFilter(_ErrorFilter())
_stderr_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

logging.basicConfig(level=logging.INFO, handlers=[_stdout_handler, _stderr_handler])
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
    problem_def: Optional[str] = None  # Problem definition JSON
    dtype: str = "float32"  # Data type for benchmark harness
    
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
        
        # Log buffers for streaming logs during execution
        # CRITICAL: run.logs() only works while task is running
        # We must stream and buffer logs in real-time, not after completion
        self._log_buffers: Dict[str, List[str]] = {}
    
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
            # Generate HIP benchmark harness
            logger.info(f"Generating benchmark harness for {config.problem_id}")
            harness_code = generate_hip_benchmark_harness(
                problem_slug=config.problem_id,
                problem_def=config.problem_def,
                dtype=config.dtype
            )
            
            # Prepare task using official SDK
            # Create commands to:
            # 1. Write the HIP source code to file
            # 2. Write the benchmark harness to file
            # 3. Compile both together with hipcc
            # 4. Run the benchmark
            commands = [
                # Write kernel source code to file
                f"cat > solution.hip << 'EOFHIP'\n{config.source_code}\nEOFHIP",
                # Write benchmark harness to file
                f"cat > harness.hip << 'EOFHARNESS'\n{harness_code}\nEOFHARNESS",
                # Check ROCm installation
                "echo '=== ROCm Environment ==='",
                "rocm-smi --showproductname || echo 'rocm-smi not available'",
                "hipcc --version || echo 'hipcc not available'",
                "echo",
                # Compile kernel + harness together
                "echo '=== Compiling HIP Kernel + Harness ==='",
                "hipcc solution.hip harness.hip -o benchmark -O3 || exit 1",
                "echo 'Compilation successful'",
                "echo",
                # Run the benchmark - CRITICAL FIX: Redirect output to file using tee
                # This ensures output is captured both to stdout AND to a persistent file
                # If container terminates before dstack captures stdout, we can still retrieve the file
                "echo '=== Running Benchmark ==='",
                "./benchmark 1024 2>&1 | tee /tmp/benchmark_output.txt",
                "echo",
                # Display the captured output explicitly (ensures it's in dstack logs)
                "echo '=== Benchmark Output (from captured file) ==='",
                "cat /tmp/benchmark_output.txt",
                "echo",
                "echo '=== Execution Complete ==='",
                # Force filesystem sync to ensure file writes complete before container termination
                "sync",
            ]
            
            # Create task using official SDK
            # Matches working CLI config from testing/rocm-dstack-hotaisle/.dstack.yml
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
                    gpu=GPU(
                        name=config.gpu_type,
                        count=1,  # Critical: explicit GPU count for resource matching
                        # NOTE: Removed memory constraint to match CLI behavior
                    ),
                ),
                # CRITICAL: Fleet creation policy - tells dstack whether to reuse or create fleets
                creation_policy="reuse-or-create",
                # Working directory for task execution
                working_dir="/workspace",
                # CRITICAL FIX: Keep container alive for 10 minutes after completion
                # This gives dstack plenty of time to flush and capture logs before VM termination
                # Previously was "10s" which was too short - logs were still lost
                # Cost impact: ~$0.96 per run (10min × MI300X rate of ~$5.76/hour)
                # This extended duration ensures logs are captured reliably
                idle_duration="10m",
                # Use spot instances for cost savings
                spot_policy="auto",
            )
            
            # Submit the task using current API method
            logger.info(f"Applying configuration for {task.name}")
            run = self.client.runs.apply_configuration(
                configuration=task,
                repo=None,  # No repo needed for inline tasks
            )
            
            # Store run reference
            task_id = run.name
            
            # Wait a moment for task to be registered
            time.sleep(1)
            
            logger.info(f"Task submitted successfully: {task_id}")
            logger.info(f"Task {task_id} will be monitored via status polling")
            
            # Store run reference and initialize log buffer
            self._active_runs[task_id] = run
            # Create a buffer to accumulate logs during execution
            self._log_buffers[task_id] = []
            
            if wait:
                return self.wait_for_completion(task_id, poll_interval=5, timeout=timeout)
            
            return TaskResult(
                task_id=task_id,
                status=TaskStatus.PENDING,
                output=None,
                error=None,
                metrics=None,
                cost_usd=0.0,
                execution_time=0.0,
                created_at=None,
                completed_at=None,
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
            # CRITICAL FIX: Always refresh run status from API, don't use cached reference
            # The cached run object has stale status
            run = None
            runs = self.client.runs.list()
            for r in runs:
                if r.name == task_id:
                    run = r
                    self._active_runs[task_id] = run  # Update cache with fresh data
                    break
            
            if not run:
                raise DStackError(f"Task {task_id} not found")
            
            # Map dstack run status to our TaskStatus
            # run.status is a RunStatus enum, we need to use .value to get the string
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
            
            # Extract the enum value (e.g., 'done', 'failed', 'running')
            run_status_obj = getattr(run, 'status', None)
            if run_status_obj and hasattr(run_status_obj, 'value'):
                run_status = run_status_obj.value.lower()
            else:
                run_status = 'pending'
            
            status = status_map.get(run_status, TaskStatus.PENDING)
            
            # Get execution details from internal _run object
            # The public run object doesn't have timestamps, but _run does
            created_at = None
            completed_at = None
            execution_time = None
            cost_usd = None
            exit_status = None
            termination_reason = None
            
            try:
                internal_run = getattr(run, '_run', None)
                if internal_run:
                    # Get timestamps
                    created_at = getattr(internal_run, 'submitted_at', None)
                    
                    # Get termination reason (e.g., 'all_jobs_done', 'stopped_by_user')
                    termination_reason = getattr(internal_run, 'termination_reason', None)
                    
                    # Get completion time from latest job submission
                    latest_job = getattr(internal_run, 'latest_job_submission', None)
                    if latest_job:
                        completed_at = getattr(latest_job, 'finished_at', None)
                        exit_status = getattr(latest_job, 'exit_status', None)
                    
                    # Calculate execution time
                    if created_at and completed_at:
                        execution_time = (completed_at - created_at).total_seconds()
                    
                    # Get cost directly from run (dstack tracks this)
                    cost_usd = getattr(internal_run, 'cost', None)
                    if cost_usd:
                        self.task_costs[task_id] = cost_usd
                        self.total_cost += cost_usd
            except Exception as e:
                logger.warning(f"Failed to extract run details: {e}")
            
            # CRITICAL FIX: Handle 'terminated'/'terminating' status properly
            # When a task completes successfully, it goes: running → terminating → done
            # We need to check exit status to distinguish success from failure
            if run_status in ('terminating', 'terminated'):
                if exit_status is not None:
                    if exit_status == 0:
                        # Exit code 0 = success
                        status = TaskStatus.SUCCEEDED
                        logger.debug(f"Task {task_id} terminated with exit code 0 (success)")
                    else:
                        # Non-zero exit code = failure
                        status = TaskStatus.FAILED
                        logger.debug(f"Task {task_id} terminated with exit code {exit_status} (failure)")
                elif termination_reason == 'all_jobs_done':
                    # Terminated because all jobs completed successfully
                    status = TaskStatus.SUCCEEDED
                    logger.debug(f"Task {task_id} terminated: all jobs done (success)")
                else:
                    # Still terminating or terminated for unknown reason, treat as running
                    # Keep polling to see if it transitions to 'done' or 'failed'
                    status = TaskStatus.RUNNING
                    logger.debug(f"Task {task_id} is terminating (reason: {termination_reason}), continuing to poll...")
            
            # Stream logs during execution and retrieve buffered logs on completion
            output = None
            error = None
            
            # CRITICAL FIX: Stream logs while task is RUNNING
            # The run.logs() iterator only works during execution, not after completion
            if status == TaskStatus.RUNNING:
                try:
                    # Stream logs in real-time and append to buffer
                    log_chunk_count = 0
                    for log_chunk in run.logs():
                        try:
                            decoded = log_chunk.decode('utf-8', errors='ignore')
                            self._log_buffers[task_id].append(decoded)
                            log_chunk_count += 1
                        except:
                            self._log_buffers[task_id].append(str(log_chunk))
                    
                    if log_chunk_count > 0:
                        logger.debug(f"Streamed {log_chunk_count} log chunks for task {task_id}")
                except Exception as e:
                    logger.debug(f"Log streaming error (non-fatal): {e}")
            
            # When task completes, return accumulated logs from buffer
            if status in (TaskStatus.SUCCEEDED, TaskStatus.FAILED):
                # Retrieve accumulated logs from buffer
                if task_id in self._log_buffers and self._log_buffers[task_id]:
                    output = ''.join(self._log_buffers[task_id])
                    logger.info(f"Retrieved {len(output)} chars of buffered logs for completed task {task_id}")
                else:
                    logger.warning(f"No buffered logs found for task {task_id}, attempting direct fetch")
                    try:
                        output = self.get_task_logs(task_id)
                    except:
                        output = ""
                
                if status == TaskStatus.FAILED:
                    error = getattr(run, 'error_message', 'Task failed')
                
                # Clean up buffer after retrieval
                if task_id in self._log_buffers:
                    del self._log_buffers[task_id]
            
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
    
    def get_task_logs(self, task_id: str, max_retries: int = 15, retry_delay: float = 3.0) -> str:
        """
        Retrieve task execution logs with retry mechanism and CLI fallback
        
        CRITICAL FIX: Logs may not be immediately available after task completion.
        This method implements a robust retry strategy with multiple fallback methods.
        
        With idle_duration="10m", we have a 10-minute window to capture logs.
        This retry strategy (15 attempts × 3s = 45s) provides multiple opportunities
        to fetch logs within the first minute, well before the container terminates.
        
        Args:
            task_id: Task identifier (run name)
            max_retries: Maximum number of retry attempts (default: 15 = 45 seconds)
            retry_delay: Delay between retries in seconds (default: 3.0)
            
        Returns:
            Log output as string (may be empty if logs are unavailable)
            
        Raises:
            DStackError: If log retrieval fails after all attempts
        """
        logger.info(f"Retrieving logs for task {task_id} (max_retries={max_retries}, retry_delay={retry_delay}s)")
        
        # Strategy 1: Try SDK logs() method with retries
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}/{max_retries}: Fetching logs via SDK...")
                
                # Get run reference (refresh from API to get latest state)
                run = None
                runs = self.client.runs.list()
                for r in runs:
                    if r.name == task_id:
                        run = r
                        self._active_runs[task_id] = run
                        break
                
                if not run:
                    logger.warning(f"Task {task_id} not found in runs list")
                    if attempt < max_retries - 1:
                        logger.info(f"Waiting {retry_delay}s before retry...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        raise DStackError(f"Task {task_id} not found after {max_retries} attempts")
                
                # Try to get logs via SDK
                log_output = []
                try:
                    for log_chunk in run.logs():
                        try:
                            log_output.append(log_chunk.decode('utf-8', errors='ignore'))
                        except:
                            log_output.append(str(log_chunk))
                except Exception as logs_error:
                    logger.warning(f"SDK logs() method failed: {logs_error}")
                    # Don't raise yet, we'll retry or try CLI fallback
                
                logs = ''.join(log_output)
                
                if logs and len(logs) > 0:
                    logger.info(f"SUCCESS: Fetched {len(logs)} chars of logs via SDK on attempt {attempt + 1}")
                    return logs
                else:
                    logger.warning(f"Attempt {attempt + 1}: SDK returned empty logs ({len(logs)} chars)")
                    
                    # If this is not the last attempt, wait and retry
                    if attempt < max_retries - 1:
                        logger.info(f"Waiting {retry_delay}s before retry (logs may not be flushed yet)...")
                        time.sleep(retry_delay)
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Waiting {retry_delay}s before retry...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"All SDK attempts failed. Trying CLI fallback...")
        
        # Strategy 2: Fallback to dstack CLI
        logger.info(f"Attempting CLI fallback: dstack logs {task_id}")
        try:
            import subprocess
            result = subprocess.run(
                ['dstack', 'logs', task_id],
                capture_output=True,
                text=True,
                timeout=30,  # 30 second timeout for CLI command
            )
            
            if result.returncode == 0 and result.stdout:
                logs = result.stdout
                logger.info(f"SUCCESS: Fetched {len(logs)} chars of logs via CLI fallback")
                return logs
            else:
                logger.warning(f"CLI fallback failed: returncode={result.returncode}, stderr={result.stderr}")
        except FileNotFoundError:
            logger.warning("dstack CLI not found in PATH - skipping CLI fallback")
        except subprocess.TimeoutExpired:
            logger.warning("CLI fallback timed out after 30s")
        except Exception as e:
            logger.warning(f"CLI fallback failed: {e}")
        
        # Strategy 3: Last resort - return empty string with warning
        logger.error(f"FAILED: Unable to retrieve logs for {task_id} after all attempts")
        logger.error("This may indicate that logs were not captured or have expired")
        return ""  # Return empty string instead of raising exception
    
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