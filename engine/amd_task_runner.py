#!/usr/bin/env python3
"""
AMD Task Runner - Standalone Python script for dstack task submission
This script is called by the Next.js API to execute AMD GPU tasks.

Accepts JSON input via stdin or command-line argument and outputs SSE-formatted events.
"""

import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dstack_runner import DStackClient, TaskConfig, TaskStatus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


def send_sse(event: str, data: Dict[str, Any]) -> None:
    """
    Send SSE event to stdout
    
    Args:
        event: Event name (matches frontend expected events)
        data: Event data as dictionary
    """
    logger.info(f"Sending SSE event: {event} - {json.dumps(data, indent=2)}")
    print(f"event: {event}", flush=True)
    print(f"data: {json.dumps(data)}", flush=True)
    print("", flush=True)


def parse_kernel_output(output: str, endpoint: str) -> Dict[str, Any]:
    """
    Parse kernel execution output to extract test results and benchmarks
    
    Args:
        output: Raw stdout from kernel execution
        endpoint: Type of endpoint (checker, benchmark, sample, sandbox)
        
    Returns:
        Parsed results dictionary
    """
    results = {
        "output": output,
        "parsed": False
    }
    
    # Parse based on endpoint type
    if endpoint == "checker":
        # Look for test results in output
        # Format: "Test 1: PASSED" or "Test 1: FAILED"
        passed_tests = 0
        total_tests = 0
        
        for line in output.split('\n'):
            if 'Test' in line and ('PASSED' in line or 'FAILED' in line):
                total_tests += 1
                if 'PASSED' in line:
                    passed_tests += 1
        
        results.update({
            "passed_tests": passed_tests,
            "total_tests": total_tests,
            "parsed": True
        })
    
    elif endpoint == "benchmark":
        # Look for benchmark results
        # Format: "Runtime: 1.234 ms" or "GFLOPS: 123.45"
        runtime_ms = None
        gflops = None
        
        for line in output.split('\n'):
            if 'Runtime:' in line and 'ms' in line:
                try:
                    runtime_ms = float(line.split('Runtime:')[1].split('ms')[0].strip())
                except:
                    pass
            if 'GFLOPS:' in line or 'GFlops:' in line:
                try:
                    gflops = float(line.split(':')[1].strip())
                except:
                    pass
        
        if runtime_ms or gflops:
            results.update({
                "runtime_ms": runtime_ms,
                "gflops": gflops,
                "parsed": True
            })
    
    return results


def execute_task(payload: Dict[str, Any]) -> int:
    """
    Execute dstack task and stream SSE events
    
    Args:
        payload: Task submission payload containing:
            - solution_code: HIP/ROCm kernel code
            - problem: Problem slug
            - problem_def: Problem definition
            - gpu_type: AMD GPU type (MI210, MI300X, etc.)
            - dtype: Data type (float32, etc.)
            - language: Programming language (hip, cuda)
            - endpoint: Endpoint type (checker, benchmark, sample, sandbox)
            - submission_id: Database submission ID (optional)
            
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        # Extract payload fields
        solution_code = payload.get('solution_code', '')
        problem = payload.get('problem', 'unknown')
        gpu_type = payload.get('gpu_type', 'MI210')
        endpoint = payload.get('endpoint', 'checker')
        
        # Use provided submission ID or generate one
        submission_id = payload.get('submission_id') or f"amd-{endpoint}-{int(time.time())}"
        
        logger.info("=" * 80)
        logger.info(f"Starting AMD task execution")
        logger.info(f"Submission ID: {submission_id}")
        logger.info(f"GPU Type: {gpu_type}")
        logger.info(f"Endpoint: {endpoint}")
        logger.info(f"Problem: {problem}")
        logger.info(f"Code length: {len(solution_code)} characters")
        logger.info("=" * 80)
        
        # Send initial queue status
        logger.info("Step 1: Sending initial queue status")
        send_sse("IN_QUEUE", {
            "status": "IN_QUEUE",
            "message": f"Submission queued for {gpu_type} GPU provisioning...",
        })
        
        # Initialize dstack client
        logger.info("Step 2: Initializing dstack client")
        client = DStackClient()
        logger.info("DStack client initialized successfully")
        
        # Verify client health
        logger.info("Step 3: Verifying dstack client health")
        if not client.health_check():
            error_msg = "dstack client health check failed"
            logger.error(f"Error occurred: {error_msg}")
            raise Exception(error_msg)
        logger.info("DStack client health check passed")
        
        send_sse("PROVISIONING", {
            "status": "PROVISIONING",
            "message": "dstack client initialized, submitting task...",
        })
        
        # Create task configuration
        logger.info("Step 4: Creating task configuration")
        config = TaskConfig(
            gpu_type=gpu_type,
            source_code=solution_code,
            problem_id=problem,
            submission_id=submission_id,
            timeout=600  # 10 minutes
        )
        logger.info("Task configuration created successfully")
        
        logger.info(f"Step 5: Submitting task to dstack for submission {submission_id}")
        
        # Submit task (non-blocking)
        result = client.submit_task(config, wait=False)
        task_id = result.task_id
        
        logger.info(f"Task submitted successfully with task ID: {task_id}")
        
        send_sse("PROVISIONING", {
            "status": "PROVISIONING",
            "message": f"Task {task_id} submitted, waiting for VM allocation...",
        })
        
        # Poll for status updates
        logger.info("Step 6: Starting status polling loop")
        last_status = None
        poll_count = 0
        max_polls = 120  # 10 minutes max (5s intervals)
        compilation_sent = False
        checking_sent = False
        
        while poll_count < max_polls:
            time.sleep(5)
            poll_count += 1
            
            try:
                result = client.get_task_status(task_id)
                
                # Send status updates when status changes
                if result.status != last_status:
                    last_status = result.status
                    logger.info(f"Status change detected: {result.status.value}")
                    
                    if result.status == TaskStatus.PROVISIONING:
                        logger.info("VM provisioning in progress")
                        send_sse("PROVISIONING", {
                            "status": "PROVISIONING",
                            "message": "VM provisioning in progress...",
                        })
                    
                    elif result.status == TaskStatus.RUNNING:
                        logger.info("Task is now running on VM")
                        if not compilation_sent:
                            logger.info("Starting compilation phase")
                            send_sse("COMPILING", {
                                "status": "COMPILING",
                                "message": "Compiling HIP kernel with hipcc...",
                            })
                            compilation_sent = True
                            time.sleep(2)
                        
                        if not checking_sent:
                            logger.info(f"Starting execution phase for endpoint: {endpoint}")
                            if endpoint == "checker":
                                send_sse("CHECKING", {
                                    "status": "CHECKING",
                                    "message": "Running test cases...",
                                })
                            elif endpoint == "benchmark":
                                send_sse("BENCHMARKING", {
                                    "status": "BENCHMARKING",
                                    "message": "Running benchmarks...",
                                })
                            else:
                                send_sse("CHECKING", {
                                    "status": "CHECKING",
                                    "message": "Executing kernel...",
                                })
                            checking_sent = True
                    
                    elif result.status == TaskStatus.SUCCEEDED:
                        logger.info("=" * 80)
                        logger.info("Task succeeded! Retrieving output...")
                        logger.info("=" * 80)
                        
                        # Get output
                        output = result.output or ""
                        logger.info(f"Output received ({len(output)} characters)")
                        
                        # Parse output based on endpoint type
                        logger.info(f"Parsing output for endpoint type: {endpoint}")
                        parsed = parse_kernel_output(output, endpoint)
                        logger.info(f"Output parsing completed: {parsed.get('parsed', False)}")
                        
                        # Send appropriate completion event
                        logger.info(f"Sending completion event for endpoint: {endpoint}")
                        if endpoint == "checker":
                            send_sse("CHECKED", {
                                "status": "CHECKED",
                                "message": "Tests completed successfully",
                                "passed_tests": parsed.get('passed_tests', 0),
                                "total_tests": parsed.get('total_tests', 1),
                                "execution_time": result.execution_time,
                                "cost_usd": result.cost_usd,
                                "output": parsed.get('output', ''),
                            })
                        elif endpoint == "benchmark":
                            send_sse("BENCHMARKED", {
                                "status": "BENCHMARKED",
                                "message": "Benchmark completed successfully",
                                "avg_runtime_ms": parsed.get('runtime_ms', 0),
                                "avg_gflops": parsed.get('gflops'),
                                "execution_time": result.execution_time,
                                "cost_usd": result.cost_usd,
                                "output": parsed.get('output', ''),
                                "submission_id": submission_id,
                            })
                        else:
                            send_sse("COMPLETED", {
                                "status": "COMPLETED",
                                "message": "Execution completed successfully",
                                "output": parsed.get('output', ''),
                                "execution_time": result.execution_time,
                                "cost_usd": result.cost_usd,
                            })
                        
                        logger.info("=" * 80)
                        logger.info("Task completed successfully - Exiting with code 0")
                        logger.info("=" * 80)
                        return 0
                    
                    elif result.status == TaskStatus.FAILED:
                        error_msg = result.error or "Task execution failed"
                        logger.error("=" * 80)
                        logger.error(f"Error occurred: Task execution failed")
                        logger.error(f"Error message: {error_msg}")
                        logger.error(f"Output: {result.output or 'No output'}")
                        logger.error("=" * 80)
                        
                        send_sse("ERROR", {
                            "status": "ERROR",
                            "error": error_msg,
                            "details": result.output or "",
                            "message": error_msg,
                            "submission_id": submission_id,
                        })
                        return 1
                    
                    elif result.status == TaskStatus.TIMEOUT:
                        logger.error("=" * 80)
                        logger.error("Error occurred: Task execution timeout")
                        logger.error("The task exceeded the maximum execution time of 10 minutes")
                        logger.error("=" * 80)
                        
                        send_sse("ERROR", {
                            "status": "ERROR",
                            "error": "Task execution timeout",
                            "details": "The task exceeded the maximum execution time of 10 minutes",
                            "message": "Execution timeout",
                            "submission_id": submission_id,
                        })
                        return 1
                    
                    elif result.status == TaskStatus.TERMINATED:
                        logger.error("=" * 80)
                        logger.error("Error occurred: Task was terminated")
                        logger.error(f"Termination reason: {result.error or 'Unknown'}")
                        logger.error("=" * 80)
                        
                        send_sse("ERROR", {
                            "status": "ERROR",
                            "error": "Task was terminated",
                            "details": result.error or "Task was manually terminated",
                            "message": "Task terminated",
                            "submission_id": submission_id,
                        })
                        return 1
                
                # Log progress every 30 seconds
                if poll_count % 6 == 0:
                    logger.info(f"Status polling in progress - Elapsed time: {poll_count * 5}s, Current status: {last_status.value if last_status else 'UNKNOWN'}")
                    
            except Exception as e:
                logger.error("=" * 80)
                logger.error(f"Error occurred during status check: {str(e)}")
                logger.error(f"Error type: {type(e).__name__}")
                logger.error("=" * 80)
                send_sse("ERROR", {
                    "status": "ERROR",
                    "error": f"Status check failed: {str(e)}",
                    "message": str(e),
                    "submission_id": submission_id,
                })
                return 1
        
        # Timeout
        logger.error("=" * 80)
        logger.error("Error occurred: Polling timeout - Task did not complete within expected time")
        logger.error(f"Total polling time: {poll_count * 5}s")
        logger.error("=" * 80)
        send_sse("ERROR", {
            "status": "ERROR",
            "error": "Task polling timeout",
            "details": "Task did not complete within expected time (10 minutes)",
            "message": "Polling timeout",
            "submission_id": submission_id,
        })
        
        # Try to terminate the task
        try:
            client.terminate_task(task_id)
        except:
            pass
        
        return 1
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error("Error occurred: Fatal error during task execution")
        logger.exception("Full exception details:")
        logger.error("=" * 80)
        send_sse("ERROR", {
            "status": "ERROR",
            "error": str(e),
            "details": str(e),
            "message": str(e),
            "submission_id": payload.get('submission_id', 'unknown'),
        })
        return 1


def main():
    """Main entry point"""
    try:
        # Read payload from command line argument or stdin
        if len(sys.argv) > 1:
            # Payload passed as command line argument
            payload_str = sys.argv[1]
        else:
            # Read from stdin
            payload_str = sys.stdin.read()
        
        # Parse JSON payload
        payload = json.loads(payload_str)
        
        # Validate required fields
        required_fields = ['solution_code', 'problem', 'gpu_type']
        missing_fields = [f for f in required_fields if f not in payload]
        
        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
        
        # Execute task
        exit_code = execute_task(payload)
        sys.exit(exit_code)
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON payload: {e}")
        send_sse("ERROR", {
            "status": "ERROR",
            "error": "Invalid JSON payload",
            "details": str(e),
        })
        sys.exit(1)
    
    except Exception as e:
        logger.exception("Fatal error")
        send_sse("ERROR", {
            "status": "ERROR",
            "error": str(e),
            "details": str(e),
        })
        sys.exit(1)


if __name__ == "__main__":
    main()