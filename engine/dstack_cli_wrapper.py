#!/usr/bin/env python3
"""
dstack CLI Wrapper for Tensara AMD GPU Orchestration

This module provides a RELIABLE interface using the dstack CLI directly.
We bypass the Python SDK entirely because it has issues with command execution.

The CLI approach is proven to work - it's what we used in successful test runs.

CRITICAL UPDATE: Now includes fleet management for new dstack versions that require
fleets to be created before task submission.
"""

import os
import sys
import json
import time
import yaml
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

# Import fleet manager for automatic fleet provisioning
FLEET_MANAGER_AVAILABLE = False
ensure_amd_fleet_ready = None

def _import_fleet_manager():
    """Dynamically import fleet manager to avoid circular imports"""
    global FLEET_MANAGER_AVAILABLE, ensure_amd_fleet_ready
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "amd_fleet_manager", 
            Path(__file__).parent / "amd_fleet_manager.py"
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            ensure_amd_fleet_ready = module.ensure_amd_fleet_ready
            FLEET_MANAGER_AVAILABLE = True
            logger.info("✅ Fleet manager imported successfully")
        else:
            raise ImportError("Failed to load fleet manager spec")
    except Exception as e:
        logger.warning(f"⚠️  Fleet manager not available: {e}")
        def ensure_amd_fleet_ready_fallback(fleet_name: Optional[str] = None) -> bool:
            logger.warning("Fleet manager not available - fleet may need to be created manually")
            return True
        ensure_amd_fleet_ready = ensure_amd_fleet_ready_fallback
        FLEET_MANAGER_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    PROVISIONING = "provisioning"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TERMINATED = "terminated"


@dataclass
class TaskResult:
    """Result of a task execution"""
    task_id: str
    status: TaskStatus
    output: Optional[str] = None
    error: Optional[str] = None
    cost_usd: Optional[float] = None
    execution_time: Optional[float] = None


class DStackCLIWrapper:
    """
    Wrapper around dstack CLI commands
    
    This is the WORKING approach - uses CLI commands directly instead of Python SDK.
    """
    
    def __init__(self):
        """Initialize the CLI wrapper"""
        self.temp_dirs: List[Path] = []
        
        # Initialize fleet manager
        _import_fleet_manager()
        
        logger.info("DStack CLI wrapper initialized")
        if FLEET_MANAGER_AVAILABLE:
            logger.info("✅ Fleet management enabled")
        else:
            logger.warning("⚠️  Fleet management disabled - manual fleet creation may be required")
    
    def submit_task(
        self,
        task_name: str,
        gpu_type: str,
        source_code: str,
        problem_id: str,
        submission_id: str,
        problem_def: str = "",
        dtype: str = "float32",
        endpoint: str = "full",
    ) -> str:
        """
        Submit a task using dstack CLI
        
        This uses the new PyTorch-based amd_remote_runner.py which:
        - Clones the problems repo from GitHub
        - Uses PyTorch ROCm for drop-in CUDA compatibility
        - Runs proper checker + benchmark phases (matching NVIDIA)
        - Outputs JSON events for frontend parsing
        
        Args:
            task_name: Name for the dstack run
            gpu_type: GPU type (e.g., MI300X)
            source_code: HIP kernel source code
            problem_id: Problem identifier (slug)
            submission_id: Submission identifier
            problem_def: Optional inline problem definition
            dtype: Data type (float32, float16, bfloat16)
            endpoint: Execution mode - "checker", "benchmark", or "full"
            
        Returns:
            Task ID (run name)
        """
        logger.info(f"Submitting task {task_name} via CLI")
        
        # CRITICAL: Ensure AMD fleet exists before task submission
        # This is required for new dstack versions
        if ensure_amd_fleet_ready:
            logger.info("🔄 Ensuring AMD DevCloud fleet is ready...")
            fleet_ready = ensure_amd_fleet_ready()
            if not fleet_ready:
                logger.error("❌ AMD fleet is not ready - task submission may fail")
                logger.error("Please check AMD DevCloud configuration and fleet status")
            else:
                logger.info("✅ AMD fleet is ready for task submission")
        else:
            logger.warning("⚠️  Fleet manager not available - proceeding without fleet validation")
        
        # Create temporary directory for this task
        temp_dir = Path(tempfile.mkdtemp(prefix=f"tensara_{submission_id}_"))
        self.temp_dirs.append(temp_dir)
        logger.info(f"Created temp directory: {temp_dir}")
        
        # Create JSON payload for the remote runner
        payload = {
            "problem": problem_id,
            "problem_def": problem_def,
            "solution_code": source_code,
            "dtype": dtype,
            "endpoint": endpoint,
            "submission_id": submission_id,
        }
        
        # Write payload to file (will be read by remote runner)
        payload_file = temp_dir / "payload.json"
        payload_file.write_text(json.dumps(payload))
        logger.info(f"Wrote payload.json ({len(source_code)} chars of code)")
        
        # Copy the remote runner script
        runner_script = Path(__file__).parent / "amd_remote_runner.py"
        if runner_script.exists():
            (temp_dir / "amd_remote_runner.py").write_text(runner_script.read_text())
            logger.info("Copied amd_remote_runner.py to temp directory")
        else:
            logger.error(f"Remote runner script not found: {runner_script}")
            raise FileNotFoundError(f"amd_remote_runner.py not found at {runner_script}")
        
        # Copy the base Problem class - required for loading problem definitions on remote VM
        # Problem definitions in tensara/problems import `from problem import Problem`,
        # but the Problem class is defined in this repo (engine/problem.py), not the problems repo
        problem_base_script = Path(__file__).parent / "problem.py"
        if problem_base_script.exists():
            (temp_dir / "problem.py").write_text(problem_base_script.read_text())
            logger.info("Copied problem.py (base Problem class) to temp directory")
        else:
            logger.error(f"Problem base class not found: {problem_base_script}")
            raise FileNotFoundError(f"problem.py not found at {problem_base_script}")
        
        # Get fleet name from environment
        fleet_name = os.getenv('AMD_FLEET_NAME', 'amd-mi300x-fleet')
        logger.info(f"Configuring task to use fleet: {fleet_name}")
        
        # Create .dstack.yml configuration
        # Uses PyTorch ROCm image with proper checker/benchmark flow
        dstack_config = {
            'type': 'task',
            'name': task_name,
            'working_dir': '/workspace',
            'idle_duration': '10m',  # Keep VM alive for 10 minutes for reuse
            # Specify fleet to run on
            'fleets': [fleet_name],
            'resources': {
                'gpu': {
                    'name': gpu_type,
                    'count': 1,
                }
            },
            # Pinned ROCm + PyTorch image for stability
            'image': 'rocm/pytorch:rocm6.2_ubuntu22.04_py3.10_pytorch_release_2.3.0',
            'env': [
                'ROCM_PATH=/opt/rocm',
                'HIP_PLATFORM=amd',
                f'GPU_TYPE={gpu_type}',
                f'PROBLEM_ID={problem_id}',
                f'SUBMISSION_ID={submission_id}',
            ],
            'files': [
                './payload.json',
                './amd_remote_runner.py',
                './problem.py',  # Base Problem class for loading problem definitions
            ],
            'commands': [
                'echo "=== ROCm Environment ==="',
                'rocm-smi --showproductname 2>/dev/null || echo "rocm-smi not available"',
                'hipcc --version 2>/dev/null || echo "hipcc not available"',
                'python3 -c "import torch; print(f\'PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}\')"',
                'echo',
                # Clone problems repo
                'echo "=== Cloning Problems Repo ==="',
                'if [ ! -d /workspace/problems ]; then git clone --depth 1 https://github.com/tensara/problems.git /workspace/problems; fi',
                'echo',
                # Run the remote runner with the payload
                'echo "=== Running AMD Remote Runner ==="',
                'python3 amd_remote_runner.py "$(cat payload.json)"',
                'echo',
                'echo "=== Execution Complete ==="',
            ],
        }
        
        # Write .dstack.yml
        config_file = temp_dir / ".dstack.yml"
        with open(config_file, 'w') as f:
            yaml.dump(dstack_config, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Wrote .dstack.yml configuration")
        
        # Submit task using dstack apply
        # CRITICAL: Use relative path to avoid macOS /var vs /private/var symlink issues
        # Use -y (skip confirmation) and -d (detach) flags
        try:
            logger.info(f"Executing: dstack apply -f .dstack.yml -y -d (in {temp_dir})")
            result = subprocess.run(
                ['dstack', 'apply', '-f', '.dstack.yml', '-y', '-d'],  # -y = no confirm, -d = detach
                cwd=str(temp_dir),  # Run from temp directory
                capture_output=True,
                text=True,
                timeout=60,
            )
            
            if result.returncode != 0:
                logger.error(f"dstack apply failed: {result.stderr}")
                raise RuntimeError(f"Failed to submit task: {result.stderr}")
            
            logger.info(f"Task submitted successfully: {task_name}")
            logger.debug(f"dstack apply output: {result.stdout}")
            
            return task_name
            
        except subprocess.TimeoutExpired:
            logger.error("dstack apply timed out after 60s")
            raise RuntimeError("Task submission timed out")
        except FileNotFoundError:
            logger.error("dstack CLI not found in PATH")
            raise RuntimeError("dstack CLI not installed or not in PATH")
        except Exception as e:
            logger.error(f"Failed to submit task: {e}")
            raise
    
    def get_task_status(self, task_name: str) -> TaskResult:
        """
        Get current status of a task using dstack ps
        
        Args:
            task_name: Task identifier (run name)
            
        Returns:
            TaskResult with current status
        """
        try:
            # Use dstack ps to get run status
            result = subprocess.run(
                ['dstack', 'ps', '--all'],
                capture_output=True,
                text=True,
                timeout=30,
            )
            
            if result.returncode != 0:
                logger.warning(f"dstack ps failed: {result.stderr}")
                return TaskResult(
                    task_id=task_name,
                    status=TaskStatus.PENDING,
                )
            
            # Parse output to find our task
            # Output format:
            # NAME                               BAC…  GPU    PRI…  STATUS      SUBMITTED    
            # tensara-cmj7oqg230000w30f6z07prir  amd…  MI30…  $1.…  exited (0)  11 hours ago 
            # Note: STATUS can be multi-word like "exited (1)" or single word like "running"
            
            lines = result.stdout.strip().split('\n')
            for line in lines[1:]:  # Skip header
                if task_name in line:
                    # Don't split - use the full line string for matching
                    line_lower = line.lower()
                    
                    # Map CLI status to our TaskStatus by checking the full line
                    if 'running' in line_lower:
                        status = TaskStatus.RUNNING
                    elif 'exited (0)' in line_lower or 'done' in line_lower:
                        status = TaskStatus.SUCCEEDED
                    elif 'exited (' in line_lower or 'failed' in line_lower:
                        # Any non-zero exit code is a failure
                        status = TaskStatus.FAILED
                    elif 'provisioning' in line_lower:
                        status = TaskStatus.PROVISIONING
                    elif 'pending' in line_lower:
                        status = TaskStatus.PENDING
                    else:
                        status = TaskStatus.PENDING
                    
                    logger.debug(f"Task {task_name} status: {status.value} (from line: {line.strip()})")
                    return TaskResult(
                        task_id=task_name,
                        status=status,
                    )
            
            # Task not found in list - might be too new or already cleaned up
            logger.debug(f"Task {task_name} not found in dstack ps output")
            return TaskResult(
                task_id=task_name,
                status=TaskStatus.PENDING,
            )
            
        except subprocess.TimeoutExpired:
            logger.warning("dstack ps timed out")
            return TaskResult(task_id=task_name, status=TaskStatus.PENDING)
        except Exception as e:
            logger.error(f"Failed to get task status: {e}")
            return TaskResult(task_id=task_name, status=TaskStatus.PENDING)
    
    def get_task_logs(self, task_name: str, max_retries: int = 10, retry_delay: float = 2.0) -> str:
        """
        Get task logs using dstack logs CLI command
        
        Args:
            task_name: Task identifier (run name)
            max_retries: Number of retry attempts
            retry_delay: Delay between retries in seconds
            
        Returns:
            Task logs as string
        """
        logger.info(f"Retrieving logs for task {task_name} via CLI")
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}/{max_retries}: Fetching logs via CLI...")
                
                result = subprocess.run(
                    ['dstack', 'logs', task_name],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                
                if result.returncode == 0 and result.stdout:
                    logs = result.stdout
                    logger.info(f"SUCCESS: Retrieved {len(logs)} characters of logs via CLI")
                    return logs
                else:
                    logger.warning(f"Attempt {attempt + 1}: CLI returned empty logs or error")
                    if attempt < max_retries - 1:
                        logger.info(f"Waiting {retry_delay}s before retry...")
                        time.sleep(retry_delay)
                
            except subprocess.TimeoutExpired:
                logger.warning(f"Attempt {attempt + 1}: CLI timed out")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
        
        # All attempts failed
        logger.error(f"Failed to retrieve logs for {task_name} after {max_retries} attempts")
        return ""
    
    def wait_for_completion(
        self,
        task_name: str,
        poll_interval: int = 5,
        timeout: int = 600,
        status_callback = None,
    ) -> TaskResult:
        """
        Wait for task to complete
        
        Args:
            task_name: Task identifier
            poll_interval: Polling interval in seconds
            timeout: Maximum wait time in seconds
            status_callback: Optional callback function called on status changes
            
        Returns:
            Final TaskResult
        """
        logger.info(f"Waiting for task {task_name} to complete (timeout={timeout}s)")
        
        start_time = time.time()
        last_status = None
        
        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                logger.error(f"Task {task_name} timed out after {timeout}s")
                return TaskResult(
                    task_id=task_name,
                    status=TaskStatus.FAILED,
                    error="Task execution timeout",
                )
            
            # Get current status
            result = self.get_task_status(task_name)
            
            # Call status callback if status changed
            if result.status != last_status:
                last_status = result.status
                logger.info(f"Task {task_name} status changed to: {result.status.value}")
                if status_callback:
                    status_callback(result.status)
            
            # Check if task is complete
            if result.status in (TaskStatus.SUCCEEDED, TaskStatus.FAILED):
                logger.info(f"Task {task_name} completed with status: {result.status.value}")
                
                # Fetch logs
                output = self.get_task_logs(task_name)
                result.output = output
                
                return result
            
            # Wait before next poll
            time.sleep(poll_interval)
    
    def terminate_task(self, task_name: str) -> bool:
        """
        Terminate a running task using dstack CLI
        
        Args:
            task_name: Task identifier (run name)
            
        Returns:
            True if termination successful
        """
        try:
            logger.info(f"Terminating task {task_name}")
            
            result = subprocess.run(
                ['dstack', 'stop', task_name],
                capture_output=True,
                text=True,
                timeout=30,
            )
            
            if result.returncode == 0:
                logger.info(f"✅ Successfully terminated task {task_name}")
                return True
            else:
                logger.warning(f"⚠️  Failed to terminate task {task_name}: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"❌ Task termination timed out for {task_name}")
            return False
        except Exception as e:
            logger.error(f"❌ Error terminating task {task_name}: {e}")
            return False
    
    def cleanup(self):
        """Clean up temporary directories"""
        for temp_dir in self.temp_dirs:
            try:
                import shutil
                shutil.rmtree(temp_dir)
                logger.debug(f"Cleaned up temp directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up {temp_dir}: {e}")
        
        self.temp_dirs.clear()
    
    def __del__(self):
        """Cleanup on destruction"""
        self.cleanup()
