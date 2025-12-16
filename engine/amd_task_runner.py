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

from dstack_cli_wrapper import DStackCLIWrapper, TaskStatus
from hip_harness import generate_hip_benchmark_harness

# Configure logging - send INFO to stdout so it doesn't appear as errors
class InfoFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == logging.INFO

class ErrorFilter(logging.Filter):
    def filter(self, record):
        return record.levelno >= logging.WARNING

# INFO and DEBUG go to stdout (normal logs)
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
stdout_handler.addFilter(InfoFilter())
stdout_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# WARNING and ERROR go to stderr (actual errors)
stderr_handler = logging.StreamHandler(sys.stderr)
stderr_handler.setLevel(logging.WARNING)
stderr_handler.addFilter(ErrorFilter())
stderr_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

logging.basicConfig(level=logging.INFO, handlers=[stdout_handler, stderr_handler])
logger = logging.getLogger(__name__)


def send_sse(event: str, data: Dict[str, Any]) -> None:
    """
    Send SSE event to stdout with SSE_EVENT marker for filtering
    
    Args:
        event: Event name (matches frontend expected events)
        data: Event data as dictionary
    """
    logger.info(f"Sending SSE event: {event} - {json.dumps(data, indent=2)}")
    # Use SSE_EVENT: prefix to distinguish from regular logs
    print(f"SSE_EVENT:event: {event}", flush=True)
    print(f"SSE_EVENT:data: {json.dumps(data)}", flush=True)
    print("SSE_EVENT:", flush=True)


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
        
        logger.info(f"Parsing benchmark output ({len(output)} chars)")
        if len(output) > 0:
            logger.info(f"Output preview (first 500 chars):\n{output[:500]}")
            logger.info(f"Output preview (last 500 chars):\n{output[-500:]}")
        else:
            logger.error("Output is completely empty - cannot parse benchmark results!")
        
        for line in output.split('\n'):
            line_lower = line.lower()
            
            # Parse runtime (case-insensitive)
            if 'runtime:' in line_lower and 'ms' in line_lower:
                try:
                    # Try multiple parsing strategies
                    if 'Runtime:' in line:
                        value_str = line.split('Runtime:')[1].split('ms')[0].strip()
                    else:
                        value_str = line_lower.split('runtime:')[1].split('ms')[0].strip()
                    runtime_ms = float(value_str)
                    logger.info(f"✓ Successfully parsed runtime: {runtime_ms} ms from line: '{line.strip()}'")
                except Exception as e:
                    logger.warning(f"✗ Failed to parse runtime from line '{line}': {e}")
            
            # Parse GFLOPS (case-insensitive)
            if 'gflops:' in line_lower or 'gflop/s:' in line_lower:
                try:
                    # Split on colon and take the value
                    parts = line.split(':')
                    if len(parts) >= 2:
                        gflops = float(parts[1].strip())
                        logger.info(f"✓ Successfully parsed GFLOPS: {gflops} from line: '{line.strip()}'")
                except Exception as e:
                    logger.warning(f"✗ Failed to parse GFLOPS from line '{line}': {e}")
        
        # Always update results for benchmark endpoint, even if parsing failed
        results.update({
            "runtime_ms": runtime_ms if runtime_ms is not None else 0.0,
            "gflops": gflops if gflops is not None else 0.0,
            "parsed": runtime_ms is not None or gflops is not None
        })
        
        if runtime_ms is None and gflops is None:
            logger.error("=" * 80)
            logger.error("PARSING FAILED: Could not extract benchmark results from output!")
            logger.error("Looking for patterns:")
            logger.error("  - 'Runtime: <number> ms' (case-insensitive)")
            logger.error("  - 'GFLOPS: <number>' (case-insensitive)")
            logger.error(f"Full output ({len(output)} chars):")
            logger.error(output if output else "(empty)")
            logger.error("=" * 80)
    
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
        problem_def = payload.get('problem_def', '')
        gpu_type = payload.get('gpu_type', 'MI210')
        endpoint = payload.get('endpoint', 'checker')
        dtype = payload.get('dtype', 'float32')
        
        # Use provided submission ID or generate one
        submission_id = payload.get('submission_id') or f"amd-{endpoint}-{int(time.time())}"
        
        logger.info("=" * 80)
        logger.info(f"Starting AMD task execution via CLI wrapper")
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
        
        # Initialize dstack CLI wrapper
        logger.info("Step 2: Initializing dstack CLI wrapper")
        client = DStackCLIWrapper()
        logger.info("DStack CLI wrapper initialized successfully")
        
        # Generate benchmark harness
        logger.info("Step 3: Generating benchmark harness")
        harness_code = generate_hip_benchmark_harness(
            problem_slug=problem,
            problem_def=problem_def,
            dtype=dtype
        )
        logger.info(f"Generated harness code: {len(harness_code)} characters")
        
        send_sse("PROVISIONING", {
            "status": "PROVISIONING",
            "message": "Submitting task to dstack...",
        })
        
        # Submit task via CLI
        logger.info(f"Step 4: Submitting task to dstack for submission {submission_id}")
        task_name = f"tensara-{submission_id}"
        
        task_id = client.submit_task(
            task_name=task_name,
            gpu_type=gpu_type,
            source_code=solution_code,
            harness_code=harness_code,
            problem_id=problem,
            submission_id=submission_id,
        )
        
        logger.info(f"Task submitted successfully with task ID: {task_id}")
        
        send_sse("PROVISIONING", {
            "status": "PROVISIONING",
            "message": f"Task {task_id} submitted, waiting for VM allocation...",
        })
        
        # Poll for status updates
        logger.info("Step 5: Starting status polling loop")
        last_status = None
        poll_count = 0
        max_polls = 120  # 10 minutes max (5s intervals)
        compilation_sent = False
        checking_sent = False
        last_heartbeat = time.time()
        HEARTBEAT_INTERVAL = 15  # Send heartbeat every 15 seconds to keep SSE connection alive
        
        while poll_count < max_polls:
            time.sleep(5)
            poll_count += 1
            
            # Send heartbeat to keep SSE connection alive during long waits
            current_time = time.time()
            if current_time - last_heartbeat >= HEARTBEAT_INTERVAL:
                send_sse("heartbeat", {"timestamp": int(current_time * 1000)})
                last_heartbeat = current_time
            
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
                        logger.info("Task succeeded! Retrieving output via CLI...")
                        logger.info("=" * 80)
                        
                        # Fetch logs using CLI
                        output = client.get_task_logs(task_id, max_retries=10, retry_delay=2.0)
                        
                        if not output or len(output) == 0:
                            logger.error("ERROR: No logs were retrieved from task!")
                            logger.error("This indicates that the task did not produce any output")
                        else:
                            logger.info(f"Successfully retrieved {len(output)} characters of logs")
                        
                        logger.info(f"Final output length: {len(output)} characters")
                        
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
                            benchmark_data = {
                                "status": "BENCHMARKED",
                                "message": "Benchmark completed successfully",
                                "avg_runtime_ms": parsed.get('runtime_ms', 0.0),
                                "avg_gflops": parsed.get('gflops', 0.0),
                                "execution_time": result.execution_time,
                                "cost_usd": result.cost_usd,
                                "output": parsed.get('output', ''),
                                "submission_id": submission_id,
                            }
                            logger.info(f"Sending BENCHMARKED event with data: runtime_ms={benchmark_data['avg_runtime_ms']}, gflops={benchmark_data['avg_gflops']}")
                            send_sse("BENCHMARKED", benchmark_data)
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
                    
                    elif result.status == TaskStatus.TERMINATED:
                        # Note: TERMINATED status should now only occur for actual failures
                        # (manual termination, killed by user, etc.) because successful
                        # terminations (exit code 0) are now mapped to SUCCEEDED in dstack_runner
                        logger.error("=" * 80)
                        logger.error("Error occurred: Task was terminated unexpectedly")
                        logger.error(f"Termination reason: {result.error or 'Unknown'}")
                        logger.error("=" * 80)
                        
                        send_sse("ERROR", {
                            "status": "ERROR",
                            "error": "Task was terminated",
                            "details": result.error or "Task was manually terminated or killed",
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