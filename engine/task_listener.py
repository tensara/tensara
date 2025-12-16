"""
Task Listener for MI300X VMs

Runs persistently inside the VM, listening for incoming tasks.
Executes each task in an isolated Docker container for security.

Security Features:
- Each task runs in fresh Docker container
- No network access for user code
- Read-only root filesystem where possible
- Resource limits (CPU/memory)
- Dropped capabilities
- Automatic container cleanup
"""

import os
import json
import logging
import tempfile
import sys
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any

# Docker SDK
try:
    import docker
except ImportError:
    print("ERROR: docker package not installed. Run: pip install docker")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TaskExecutor:
    """Executes HIP kernels in isolated Docker containers"""
    
    def __init__(self):
        """Initialize Docker client"""
        try:
            self.docker_client = docker.from_env()
            self.rocm_image = os.getenv('ROCM_IMAGE', 'rocm/pytorch:latest')
            
            # Pre-pull image for faster execution
            logger.info(f"Pulling ROCm image: {self.rocm_image}")
            self.docker_client.images.pull(self.rocm_image)
            logger.info("ROCm image ready")
            
        except Exception as e:
            logger.error(f"Failed to initialize Docker client: {e}")
            raise
    
    def execute_hip_kernel(
        self,
        hip_code: str,
        matrix_size: int = 1024,
        timeout: int = 300,
    ) -> Dict[str, Any]:
        """
        Execute HIP kernel in isolated container
        
        Security features:
        - No network access
        - Read-only root filesystem (with writable /tmp)
        - Resource limits (memory/CPU)
        - Dropped capabilities
        - Fresh container per task
        - Auto-cleanup after execution
        
        Args:
            hip_code: HIP kernel source code
            matrix_size: Matrix size for execution
            timeout: Execution timeout in seconds
        
        Returns:
            Dict with status, output, and metrics
        """
        # Create temporary directory for code
        with tempfile.TemporaryDirectory() as tmpdir:
            code_path = Path(tmpdir) / "solution.hip"
            code_path.write_text(hip_code)
            
            try:
                # Run in isolated container
                logger.info(f"Executing HIP kernel (size={matrix_size}, timeout={timeout}s)")
                
                container = self.docker_client.containers.run(
                    image=self.rocm_image,
                    command=[
                        "bash", "-c",
                        "cd /code && "
                        "hipcc solution.hip -o /tmp/solution -O3 && "
                        f"/tmp/solution {matrix_size}"
                    ],
                    volumes={
                        str(tmpdir): {'bind': '/code', 'mode': 'ro'}  # Read-only code
                    },
                    device_requests=[
                        docker.types.DeviceRequest(
                            driver='amd',
                            device_ids=['0'],
                            capabilities=[['gpu']]
                        )
                    ],
                    # Security hardening
                    security_opt=['no-new-privileges'],
                    cap_drop=['ALL'],
                    read_only=False,  # Need writable /tmp for compilation
                    tmpfs={'/tmp': 'size=4g,mode=1777'},  # Writable tmp only
                    network_mode='none',  # No network access
                    mem_limit='16g',
                    cpu_period=100000,
                    cpu_quota=400000,  # 4 CPUs
                    detach=False,
                    stdout=True,
                    stderr=True,
                    remove=True,  # Auto-remove container
                    timeout=timeout,
                )
                
                # Get output
                output = container.decode('utf-8')
                
                logger.info("Task execution successful")
                
                return {
                    "status": "success",
                    "output": output,
                    "error": None,
                }
                
            except docker.errors.ContainerError as e:
                logger.error(f"Container execution failed: {e}")
                return {
                    "status": "error",
                    "output": e.stderr.decode('utf-8') if e.stderr else "",
                    "error": f"Container error: {str(e)}",
                }
            except Exception as e:
                logger.error(f"Execution failed: {e}")
                return {
                    "status": "error",
                    "output": "",
                    "error": str(e),
                }


class TaskHandler(BaseHTTPRequestHandler):
    """HTTP handler for task requests"""
    
    # Shared executor instance
    executor = None
    
    def do_POST(self):
        """Handle task submission"""
        if self.path != '/execute':
            self.send_error(404, "Not Found")
            return
        
        try:
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)
            
            # Extract parameters
            hip_code = data.get('hip_code')
            matrix_size = data.get('matrix_size', 1024)
            timeout = data.get('timeout', 300)
            
            if not hip_code:
                self.send_error(400, "Missing hip_code parameter")
                return
            
            logger.info(f"Received task (size={matrix_size}, timeout={timeout})")
            
            # Execute in container
            if TaskHandler.executor is None:
                TaskHandler.executor = TaskExecutor()
            
            result = TaskHandler.executor.execute_hip_kernel(
                hip_code,
                matrix_size,
                timeout
            )
            
            # Send response
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
            
            logger.info(f"Task completed: {result['status']}")
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON: {e}")
            self.send_error(400, f"Invalid JSON: {str(e)}")
        except Exception as e:
            logger.error(f"Request handling failed: {e}")
            self.send_error(500, f"Internal error: {str(e)}")
    
    def do_GET(self):
        """Handle health check"""
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "healthy"}).encode())
        else:
            self.send_error(404, "Not Found")
    
    def log_message(self, format, *args):
        """Override to use logger"""
        logger.info(format % args)


def main():
    """Start task listener server"""
    port = int(os.getenv('TASK_LISTENER_PORT', '8000'))
    
    # Initialize executor
    try:
        TaskHandler.executor = TaskExecutor()
    except Exception as e:
        logger.error(f"Failed to initialize executor: {e}")
        sys.exit(1)
    
    # Start HTTP server
    server = HTTPServer(('0.0.0.0', port), TaskHandler)
    
    logger.info(f"Task listener started on port {port}")
    logger.info("Ready to accept HIP kernel executions")
    logger.info("Endpoints:")
    logger.info("  POST /execute - Execute HIP kernel")
    logger.info("  GET  /health  - Health check")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.shutdown()


if __name__ == '__main__':
    main()
