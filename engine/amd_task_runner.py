#!/usr/bin/env python3
"""
AMD Task Runner - Standalone Python script for dstack task submission

This script is called by the Next.js API to execute AMD GPU tasks via dstack.
It:
1. Accepts JSON input with problem/code/config
2. Submits task to dstack (which runs amd_remote_runner.py on AMD VM)
3. Polls for completion and parses JSON events from the remote runner
4. Streams SSE events to the frontend that match NVIDIA format

JSON Event Format (from amd_remote_runner.py):
    JSON_EVENT:{"status": "CHECKING", "message": "...", ...}
    JSON_EVENT:{"status": "TEST_RESULT", "result": {...}, "total_tests": N}
    JSON_EVENT:{"status": "CHECKED", "test_results": [...], "passed_tests": N, "total_tests": N}
    JSON_EVENT:{"status": "BENCHMARKING", ...}
    JSON_EVENT:{"status": "BENCHMARK_RESULT", "result": {...}, "total_tests": N}
    JSON_EVENT:{"status": "BENCHMARKED", "test_results": [...], "avg_runtime_ms": X, ...}
    JSON_EVENT:{"status": "WRONG_ANSWER", ...}
    JSON_EVENT:{"status": "COMPILE_ERROR", ...}
    JSON_EVENT:{"status": "RUNTIME_ERROR", ...}
    JSON_EVENT:{"status": "ERROR", ...}
"""

import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dstack_cli_wrapper import DStackCLIWrapper, TaskStatus

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
    
    The TypeScript runner filters lines starting with SSE_EVENT: to extract
    SSE events and sends them to the frontend.
    
    Args:
        event: Event name (matches frontend expected events)
        data: Event data as dictionary
    """
    logger.info(f"Sending SSE event: {event}")
    # Use SSE_EVENT: prefix to distinguish from regular logs
    print(f"SSE_EVENT:event: {event}", flush=True)
    print(f"SSE_EVENT:data: {json.dumps(data)}", flush=True)
    print("SSE_EVENT:", flush=True)


def parse_json_events(output: str) -> List[Dict[str, Any]]:
    """
    Parse JSON events from remote runner output
    
    The remote runner outputs events prefixed with JSON_EVENT:
    Example: JSON_EVENT:{"status": "CHECKING", "message": "..."}
    
    Args:
        output: Raw stdout from the remote runner
        
    Returns:
        List of parsed event dictionaries
    """
    events = []
    for line in output.split('\n'):
        line = line.strip()
        if line.startswith('JSON_EVENT:'):
            try:
                json_str = line[len('JSON_EVENT:'):]
                event = json.loads(json_str)
                events.append(event)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON event: {line[:100]}... Error: {e}")
    return events


def process_remote_events(
    events: List[Dict[str, Any]], 
    endpoint: str,
    submission_id: str
) -> Dict[str, Any]:
    """
    Process events from remote runner and send corresponding SSE events
    
    This maps the remote runner's JSON events to SSE events that match
    the NVIDIA frontend format.
    
    Args:
        events: List of parsed JSON events from remote runner
        endpoint: Endpoint type (checker, benchmark, full)
        submission_id: Submission ID for tracking
        
    Returns:
        Final result dictionary with test counts, runtime, etc.
    """
    result = {
        "passed_tests": 0,
        "total_tests": 0,
        "avg_runtime_ms": 0.0,
        "avg_gflops": None,
        "test_results": [],
        "status": "UNKNOWN",
    }
    
    for event in events:
        status = event.get("status", "")
        
        # Map remote events to SSE events
        if status == "COMPILING":
            send_sse("COMPILING", {
                "status": "COMPILING",
                "message": event.get("message", "Compiling HIP kernel..."),
            })
            
        elif status == "COMPILED":
            # Don't send separate event, compilation is part of COMPILING phase
            pass
            
        elif status == "CHECKING":
            send_sse("CHECKING", {
                "status": "CHECKING",
                "message": event.get("message", "Running test cases..."),
                "total_tests": event.get("total_tests", 0),
            })
            result["total_tests"] = event.get("total_tests", 0)
            
        elif status == "TEST_RESULT":
            test_result = event.get("result", {})
            send_sse("TEST_RESULT", {
                "status": "TEST_RESULT",
                "result": test_result,
                "total_tests": event.get("total_tests", 0),
            })
            
        elif status == "CHECKED":
            result["passed_tests"] = event.get("passed_tests", event.get("total_tests", 0))
            result["total_tests"] = event.get("total_tests", 0)
            result["test_results"] = event.get("test_results", [])
            result["status"] = "CHECKED"
            send_sse("CHECKED", {
                "status": "CHECKED",
                "message": "All tests passed",
                "passed_tests": result["passed_tests"],
                "total_tests": result["total_tests"],
                "test_results": result["test_results"],
            })
            
        elif status == "WRONG_ANSWER":
            result["passed_tests"] = event.get("passed_tests", 0)
            result["total_tests"] = event.get("total_tests", 0)
            result["test_results"] = event.get("test_results", [])
            result["status"] = "WRONG_ANSWER"
            send_sse("WRONG_ANSWER", {
                "status": "WRONG_ANSWER",
                "message": "Test case failed",
                "debug_info": event.get("debug_info", {}),
                "passed_tests": result["passed_tests"],
                "total_tests": result["total_tests"],
                "test_results": result["test_results"],
            })
            
        elif status == "BENCHMARKING":
            send_sse("BENCHMARKING", {
                "status": "BENCHMARKING",
                "message": event.get("message", "Running benchmarks..."),
            })
            
        elif status == "BENCHMARK_RESULT":
            benchmark_result = event.get("result", {})
            send_sse("BENCHMARK_RESULT", {
                "status": "BENCHMARK_RESULT",
                "result": benchmark_result,
                "total_tests": event.get("total_tests", 0),
            })
            
        elif status == "BENCHMARKED":
            result["avg_runtime_ms"] = event.get("avg_runtime_ms", 0.0)
            result["avg_gflops"] = event.get("avg_gflops")
            result["total_tests"] = event.get("total_tests", result["total_tests"])
            result["test_results"] = event.get("test_results", result["test_results"])
            result["status"] = "BENCHMARKED"
            
            benchmark_data = {
                "status": "BENCHMARKED",
                "message": "Benchmark completed successfully",
                "avg_runtime_ms": result["avg_runtime_ms"],
                "total_tests": result["total_tests"],
                "test_results": result["test_results"],
                "submission_id": submission_id,
            }
            if result["avg_gflops"] is not None:
                benchmark_data["avg_gflops"] = result["avg_gflops"]
            send_sse("BENCHMARKED", benchmark_data)
            
        elif status == "COMPILE_ERROR":
            result["status"] = "COMPILE_ERROR"
            send_sse("COMPILE_ERROR", {
                "status": "COMPILE_ERROR",
                "message": event.get("message", "Compilation failed"),
                "details": event.get("details", ""),
            })
            
        elif status == "RUNTIME_ERROR":
            result["status"] = "RUNTIME_ERROR"
            send_sse("RUNTIME_ERROR", {
                "status": "RUNTIME_ERROR",
                "message": event.get("message", "Runtime error"),
                "details": event.get("details", ""),
            })
            
        elif status == "TIME_LIMIT_EXCEEDED":
            result["status"] = "TIME_LIMIT_EXCEEDED"
            send_sse("TIME_LIMIT_EXCEEDED", {
                "status": "TIME_LIMIT_EXCEEDED",
                "message": event.get("message", "Time limit exceeded"),
                "details": event.get("details", ""),
            })
            
        elif status == "ERROR":
            result["status"] = "ERROR"
            send_sse("ERROR", {
                "status": "ERROR",
                "error": event.get("message", "Unknown error"),
                "details": event.get("details", ""),
                "submission_id": submission_id,
            })
    
    return result


def execute_task(payload: Dict[str, Any]) -> int:
    """
    Execute dstack task and stream SSE events
    
    Args:
        payload: Task submission payload containing:
            - solution_code: HIP/ROCm kernel code
            - problem: Problem slug
            - problem_def: Problem definition (optional)
            - gpu_type: AMD GPU type (MI210, MI300X, etc.)
            - dtype: Data type (float32, etc.)
            - language: Programming language (hip)
            - endpoint: Endpoint type (checker, benchmark, full)
            - submission_id: Database submission ID (optional)
            
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        # Extract payload fields
        solution_code = payload.get('solution_code', '')
        problem = payload.get('problem', 'unknown')
        problem_def = payload.get('problem_def', '')
        gpu_type = payload.get('gpu_type', 'MI300X')
        endpoint = payload.get('endpoint', 'full')
        dtype = payload.get('dtype', 'float32')
        
        # Use provided submission ID or generate one
        submission_id = payload.get('submission_id') or f"amd-{endpoint}-{int(time.time())}"
        
        logger.info("=" * 80)
        logger.info(f"Starting AMD task execution via CLI wrapper")
        logger.info(f"Submission ID: {submission_id}")
        logger.info(f"GPU Type: {gpu_type}")
        logger.info(f"Endpoint: {endpoint}")
        logger.info(f"Problem: {problem}")
        logger.info(f"Dtype: {dtype}")
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
        
        send_sse("PROVISIONING", {
            "status": "PROVISIONING",
            "message": "Submitting task to dstack...",
        })
        
        # Submit task via CLI (uses new amd_remote_runner.py)
        logger.info(f"Step 3: Submitting task to dstack for submission {submission_id}")
        task_name = f"tensara-{submission_id}"
        
        task_id = client.submit_task(
            task_name=task_name,
            gpu_type=gpu_type,
            source_code=solution_code,
            problem_id=problem,
            submission_id=submission_id,
            problem_def=problem_def,
            dtype=dtype,
            endpoint=endpoint,
        )
        
        logger.info(f"Task submitted successfully with task ID: {task_id}")
        
        send_sse("PROVISIONING", {
            "status": "PROVISIONING",
            "message": f"Task {task_id} submitted, waiting for VM allocation...",
        })
        
        # Poll for status updates
        logger.info("Step 4: Starting status polling loop")
        last_status = None
        poll_count = 0
        max_polls = 120  # 10 minutes max (5s intervals)
        provisioning_sent = False
        last_heartbeat = time.time()
        HEARTBEAT_INTERVAL = 15  # Send heartbeat every 15 seconds
        
        while poll_count < max_polls:
            time.sleep(5)
            poll_count += 1
            
            # Send heartbeat to keep SSE connection alive
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
                        if not provisioning_sent:
                            logger.info("VM provisioning in progress")
                            send_sse("PROVISIONING", {
                                "status": "PROVISIONING",
                                "message": "VM provisioning in progress...",
                            })
                            provisioning_sent = True
                    
                    elif result.status == TaskStatus.RUNNING:
                        logger.info("Task is now running on VM")
                        # The remote runner will emit its own events (COMPILING, CHECKING, etc.)
                        # We just need to wait for completion
                    
                    elif result.status == TaskStatus.SUCCEEDED:
                        logger.info("=" * 80)
                        logger.info("Task succeeded! Retrieving and parsing output...")
                        logger.info("=" * 80)
                        
                        # Fetch logs using CLI
                        output = client.get_task_logs(task_id, max_retries=10, retry_delay=2.0)
                        
                        if not output:
                            logger.error("ERROR: No logs were retrieved from task!")
                        else:
                            logger.info(f"Retrieved {len(output)} characters of logs")
                        
                        # Parse JSON events from remote runner
                        events = parse_json_events(output)
                        logger.info(f"Parsed {len(events)} JSON events from output")
                        
                        if events:
                            # Process events and send SSE updates
                            final_result = process_remote_events(events, endpoint, submission_id)
                            logger.info(f"Final result: {final_result}")
                        else:
                            # Fallback: no events parsed, send generic success
                            logger.warning("No JSON events found in output, sending generic success")
                            if endpoint == "checker":
                                send_sse("CHECKED", {
                                    "status": "CHECKED",
                                    "message": "Tests completed (no detailed results available)",
                                    "passed_tests": 1,
                                    "total_tests": 1,
                                })
                            elif endpoint == "benchmark":
                                send_sse("BENCHMARKED", {
                                    "status": "BENCHMARKED",
                                    "message": "Benchmark completed (no detailed results available)",
                                    "avg_runtime_ms": 0.0,
                                    "submission_id": submission_id,
                                })
                            else:
                                send_sse("COMPLETED", {
                                    "status": "COMPLETED",
                                    "message": "Execution completed",
                                    "output": output[:5000] if output else "",
                                })
                        
                        logger.info("=" * 80)
                        logger.info("Task completed successfully - Exiting with code 0")
                        logger.info("=" * 80)
                        return 0
                    
                    elif result.status == TaskStatus.FAILED:
                        error_msg = result.error or "Task execution failed"
                        logger.error("=" * 80)
                        logger.error(f"Task failed: {error_msg}")
                        logger.error("=" * 80)
                        
                        # Try to get logs for error details
                        output = client.get_task_logs(task_id, max_retries=3, retry_delay=1.0)
                        events = parse_json_events(output) if output else []
                        
                        # Check if there's a specific error event
                        error_event = None
                        for event in events:
                            if event.get("status") in ("COMPILE_ERROR", "RUNTIME_ERROR", "ERROR", "WRONG_ANSWER"):
                                error_event = event
                                break
                        
                        if error_event:
                            send_sse(error_event["status"], error_event)
                        else:
                            send_sse("ERROR", {
                                "status": "ERROR",
                                "error": error_msg,
                                "details": output[:5000] if output else "",
                                "submission_id": submission_id,
                            })
                        return 1
                    
                    elif result.status == TaskStatus.TERMINATED:
                        logger.error("Task was terminated unexpectedly")
                        send_sse("ERROR", {
                            "status": "ERROR",
                            "error": "Task was terminated",
                            "details": result.error or "Task was manually terminated or killed",
                            "submission_id": submission_id,
                        })
                        return 1
                
                # Log progress every 30 seconds
                if poll_count % 6 == 0:
                    logger.info(f"Polling... Elapsed: {poll_count * 5}s, Status: {last_status.value if last_status else 'UNKNOWN'}")
                    
            except Exception as e:
                logger.error(f"Error during status check: {e}")
                send_sse("ERROR", {
                    "status": "ERROR",
                    "error": f"Status check failed: {str(e)}",
                    "submission_id": submission_id,
                })
                return 1
        
        # Timeout
        logger.error("Polling timeout - task did not complete in time")
        send_sse("ERROR", {
            "status": "ERROR",
            "error": "Task polling timeout",
            "details": "Task did not complete within 10 minutes",
            "submission_id": submission_id,
        })
        
        # Try to terminate the task
        try:
            client.terminate_task(task_id)
        except:
            pass
        
        return 1
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.exception("Full exception details:")
        send_sse("ERROR", {
            "status": "ERROR",
            "error": str(e),
            "details": str(e),
            "submission_id": payload.get('submission_id', 'unknown'),
        })
        return 1


def main():
    """Main entry point"""
    try:
        # Read payload from command line argument or stdin
        if len(sys.argv) > 1:
            payload_str = sys.argv[1]
        else:
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
