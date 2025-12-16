"""
VM Orchestrator for AMD DevCloud MI300X Instances

Manages VM lifecycle:
- Provision VMs on-demand (cold start: 2-5 minutes)
- Reuse warm VMs across users (< 10 seconds)
- Automatic idle shutdown (10 minutes)
- Cost tracking with $1,400 grant
- Docker container isolation per task

Architecture:
- Single GPU type: MI300X @ $1.99/hour
- 10-minute idle timeout
- Cross-user VM reuse (aggressive policy)
- Thread-safe VM pool management
"""

import os
import time
import json
import logging
import hashlib
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

# Import existing dstack runner
from dstack_runner import DStackClient, TaskConfig, TaskResult, TaskStatus

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VMState(Enum):
    """VM lifecycle states"""
    COLD = "cold"                    # No VM exists
    PROVISIONING = "provisioning"    # VM being created (2-5 min)
    WARMING = "warming"              # GPU warmup in progress (~30s)
    WARM = "warm"                    # Ready to accept tasks
    BUSY = "busy"                    # Task currently executing
    IDLE = "idle"                    # Waiting for next task (idle timer running)
    SHUTTING_DOWN = "shutting_down"  # VM terminating


@dataclass
class VMInstance:
    """Represents a managed MI300X VM instance"""
    id: str  # Unique VM identifier
    dstack_run_id: str  # dstack run ID
    gpu_type: str  # Always "MI300X" for now
    state: VMState
    created_at: datetime
    last_used_at: datetime
    idle_since: Optional[datetime] = None
    total_cost_usd: float = 0.0
    tasks_completed: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'dstack_run_id': self.dstack_run_id,
            'gpu_type': self.gpu_type,
            'state': self.state.value,
            'created_at': self.created_at.isoformat(),
            'last_used_at': self.last_used_at.isoformat(),
            'idle_since': self.idle_since.isoformat() if self.idle_since else None,
            'total_cost_usd': round(self.total_cost_usd, 2),
            'tasks_completed': self.tasks_completed,
        }


class VMOrchestrator:
    """
    Orchestrates MI300X VMs on AMD DevCloud
    
    Features:
    - Single GPU type: MI300X @ $1.99/hour
    - 10-minute idle timeout
    - Cross-user VM reuse
    - Docker container isolation
    - Cost tracking with grant credits
    """
    
    def __init__(self):
        """Initialize orchestrator"""
        # Configuration from environment
        self.idle_timeout = int(os.getenv('AMD_VM_IDLE_TIMEOUT', '600'))  # 10 min
        self.max_concurrent_vms = int(os.getenv('AMD_VM_MAX_CONCURRENT', '3'))
        self.mi300x_hourly_rate = float(os.getenv('AMD_MI300X_HOURLY_RATE', '1.99'))
        self.grant_credits_total = float(os.getenv('AMD_GRANT_CREDITS_TOTAL', '1400.00'))
        self.cost_alert_threshold = float(os.getenv('AMD_COST_ALERT_THRESHOLD', '0.8'))
        
        # VM pool (key: gpu_type, value: list of VMs)
        # Using a lock for thread-safe operations
        self.vm_pool: Dict[str, List[VMInstance]] = {"MI300X": []}
        self.vm_pool_lock = threading.Lock()
        
        # Cost tracking
        self.total_cost_usd = 0.0
        self.tasks_completed = 0
        self.cold_starts = 0
        self.warm_reuses = 0
        
        # dstack client
        self.dstack_client = DStackClient()
        
        # Start background cleanup thread
        self.cleanup_thread = threading.Thread(
            target=self._background_cleanup,
            daemon=True
        )
        self.cleanup_thread.start()
        
        logger.info("VM Orchestrator initialized")
        logger.info(f"Config: idle_timeout={self.idle_timeout}s, "
                   f"max_vms={self.max_concurrent_vms}, "
                   f"rate=${self.mi300x_hourly_rate}/hr")
    
    def get_or_create_vm(self, gpu_type: str = "MI300X") -> VMInstance:
        """
        Get existing warm VM or provision new one
        
        Returns:
            VMInstance ready to accept tasks
        
        Raises:
            ValueError: If unsupported GPU type
            Exception: If VM limit reached or provisioning fails
        """
        # Only support MI300X for now
        if gpu_type != "MI300X":
            raise ValueError(f"Only MI300X supported, got: {gpu_type}")
        
        with self.vm_pool_lock:
            # Try to find WARM or IDLE VM
            vm = self._find_available_vm(gpu_type)
            
            if vm:
                logger.info(f"Reusing warm VM: {vm.id}")
                vm.state = VMState.BUSY
                vm.last_used_at = datetime.utcnow()
                vm.idle_since = None
                self.warm_reuses += 1
                return vm
            
            # No warm VM available - provision new one
            logger.info(f"No warm VM available, provisioning new {gpu_type}...")
            vm = self._provision_new_vm(gpu_type)
            self.cold_starts += 1
            return vm
    
    def _find_available_vm(self, gpu_type: str) -> Optional[VMInstance]:
        """Find WARM or IDLE VM (must be called within lock)"""
        for vm in self.vm_pool.get(gpu_type, []):
            if vm.state in (VMState.WARM, VMState.IDLE):
                return vm
        return None
    
    def _provision_new_vm(self, gpu_type: str) -> VMInstance:
        """
        Provision new MI300X VM via dstack (must be called within lock)
        
        Steps:
        1. Check VM limit
        2. Submit dstack task with idle_duration=10m
        3. Wait for VM ready
        4. Run GPU warmup
        5. Add to pool as WARM
        """
        # Check VM limit
        active_vms = len([vm for vm in self.vm_pool.get(gpu_type, [])
                         if vm.state not in (VMState.SHUTTING_DOWN,)])
        if active_vms >= self.max_concurrent_vms:
            raise Exception(f"Max concurrent VMs reached ({self.max_concurrent_vms})")
        
        # Create VM instance
        vm_id = f"mi300x-{int(time.time())}"
        vm = VMInstance(
            id=vm_id,
            dstack_run_id="",  # Will be set after provision
            gpu_type=gpu_type,
            state=VMState.PROVISIONING,
            created_at=datetime.utcnow(),
            last_used_at=datetime.utcnow(),
        )
        
        # Add to pool
        self.vm_pool[gpu_type].append(vm)
        
        try:
            # Submit warmup task to dstack
            # This provisions VM and keeps it warm with idle_duration
            task_config = TaskConfig(
                gpu_type=gpu_type,
                source_code=self._get_warmup_script(),
                problem_id="warmup",
                submission_id=vm_id,
                timeout=600,  # 10 min
            )
            
            logger.info(f"Submitting warmup task for VM {vm_id}...")
            result = self.dstack_client.submit_task(task_config, wait=True)
            
            if result.status == TaskStatus.SUCCEEDED:
                vm.dstack_run_id = result.task_id
                vm.state = VMState.WARM
                logger.info(f"VM {vm_id} provisioned successfully (dstack_run: {result.task_id})")
                return vm
            else:
                vm.state = VMState.SHUTTING_DOWN
                raise Exception(f"VM provisioning failed: {result.error}")
                
        except Exception as e:
            logger.error(f"Failed to provision VM: {e}")
            vm.state = VMState.SHUTTING_DOWN
            raise
    
    def _get_warmup_script(self) -> str:
        """Generate GPU warmup script"""
        return """
#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void warmup_kernel(float* data, int n) {
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f + 1.0f;
    }
}

int main() {
    printf("=== MI300X GPU Warmup ===\\n");
    
    int n = 1024 * 1024;
    size_t bytes = n * sizeof(float);
    
    float *d_data;
    hipMalloc(&d_data, bytes);
    
    dim3 threads(256);
    dim3 blocks((n + threads.x - 1) / threads.x);
    
    hipLaunchKernelGGL(warmup_kernel, blocks, threads, 0, 0, d_data, n);
    hipDeviceSynchronize();
    
    hipFree(d_data);
    
    printf("GPU warmup complete\\n");
    return 0;
}
"""
    
    def submit_task(self, vm: VMInstance, task_config: TaskConfig) -> TaskResult:
        """
        Submit task to VM with Docker isolation
        
        Args:
            vm: Target VM instance
            task_config: Task configuration
            
        Returns:
            TaskResult with execution results
        """
        if vm.state != VMState.BUSY:
            raise ValueError(f"VM not ready: {vm.state}")
        
        try:
            # Submit task to dstack (will run on warm VM)
            result = self.dstack_client.submit_task(task_config, wait=True)
            
            with self.vm_pool_lock:
                # Update VM state
                vm.tasks_completed += 1
                vm.last_used_at = datetime.utcnow()
                vm.state = VMState.IDLE
                vm.idle_since = datetime.utcnow()
                
                # Update cost
                if result.execution_time:
                    cost = (result.execution_time / 3600.0) * self.mi300x_hourly_rate
                    vm.total_cost_usd += cost
                    self.total_cost_usd += cost
                
                self.tasks_completed += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            with self.vm_pool_lock:
                vm.state = VMState.IDLE
                vm.idle_since = datetime.utcnow()
            raise
    
    def _background_cleanup(self):
        """Background thread to shutdown idle VMs"""
        while True:
            try:
                time.sleep(60)  # Check every minute
                self._shutdown_idle_vms()
                self._check_credit_usage()
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
    
    def _shutdown_idle_vms(self):
        """Shutdown VMs idle > threshold"""
        now = datetime.utcnow()
        
        with self.vm_pool_lock:
            for gpu_type, vms in self.vm_pool.items():
                for vm in vms[:]:  # Copy list to allow removal during iteration
                    if vm.state == VMState.IDLE and vm.idle_since:
                        idle_duration = (now - vm.idle_since).total_seconds()
                        
                        if idle_duration >= self.idle_timeout:
                            logger.info(f"Shutting down idle VM {vm.id} "
                                       f"(idle for {idle_duration:.0f}s)")
                            self._shutdown_vm(vm)
    
    def _shutdown_vm(self, vm: VMInstance):
        """Terminate VM (must be called within lock)"""
        try:
            vm.state = VMState.SHUTTING_DOWN
            
            # Terminate via dstack
            if vm.dstack_run_id:
                self.dstack_client.terminate_task(vm.dstack_run_id)
            
            # Remove from pool
            self.vm_pool[vm.gpu_type].remove(vm)
            
            logger.info(f"VM {vm.id} terminated "
                       f"(tasks={vm.tasks_completed}, cost=${vm.total_cost_usd:.2f})")
            
        except Exception as e:
            logger.error(f"Failed to shutdown VM {vm.id}: {e}")
    
    def _check_credit_usage(self):
        """Alert when credits reach threshold"""
        if self.grant_credits_total > 0:
            usage_pct = self.total_cost_usd / self.grant_credits_total
            
            if usage_pct >= self.cost_alert_threshold:
                logger.warning(f"⚠️  AMD DevCloud credits at {usage_pct*100:.1f}%")
                logger.warning(f"Used: ${self.total_cost_usd:.2f} / ${self.grant_credits_total:.2f}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get orchestrator metrics"""
        with self.vm_pool_lock:
            active_vms = sum(
                len([vm for vm in vms if vm.state not in (VMState.SHUTTING_DOWN,)])
                for vms in self.vm_pool.values()
            )
            
            return {
                "total_cost_usd": round(self.total_cost_usd, 2),
                "remaining_credits_usd": round(self.grant_credits_total - self.total_cost_usd, 2),
                "credits_used_pct": round((self.total_cost_usd / self.grant_credits_total) * 100, 1) if self.grant_credits_total > 0 else 0,
                "tasks_completed": self.tasks_completed,
                "cold_starts": self.cold_starts,
                "warm_reuses": self.warm_reuses,
                "active_vms": active_vms,
                "vm_pool": {
                    gpu_type: [vm.to_dict() for vm in vms]
                    for gpu_type, vms in self.vm_pool.items()
                },
            }
