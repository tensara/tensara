# please dont look here
# this is a MESS
import modal
import tempfile
import os
from pathlib import Path
from fastapi.responses import StreamingResponse
import json

stub_dir = Path(__file__).parent
image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install([
        "build-essential",
        "make",
        "python3-dev", 
        "python3-pip",
        "g++"
    ])
    .pip_install([
        "torch",
        "ninja",
        "fastapi[standard]"
    ])
    .env({"CXX": "g++"})
    .add_local_file(stub_dir / "benchmark/benchmark.cu", "/root/benchmark.cu")
    .add_local_file(stub_dir / "benchmark/core.hpp", "/root/core.hpp")
    .add_local_file(stub_dir / "benchmark/Makefile", "/root/Makefile")
    .add_local_file(stub_dir / "checker/checker.cu", "/root/checker.cu")
    .add_local_file(stub_dir / "checker/core.hpp", "/root/checker/core.hpp")
    .add_local_file(stub_dir / "checker/tests.hpp", "/root/checker/tests.hpp")
    .add_local_file(stub_dir / "checker/Makefile", "/root/checker/Makefile")
)

app = modal.App("tensara", image=image)

async def generic_checker(item: dict):
    """Common implementation for all checker endpoints."""
    async def generate_checker_results():
        import subprocess
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                yield "data: " + json.dumps({"status": "compiling"}) + "\n\n"
                
                tmpdir_path = Path(tmpdir)
                
                checker_dir = tmpdir_path / "checker"
                checker_dir.mkdir()
                os.system(f"cp /root/checker/core.hpp /root/checker/Makefile {str(checker_dir)}")
                
                solution_path = checker_dir / "solution.cu"
                solution_path.write_text(item["solution_code"])
                
                tests_path = checker_dir / "tests.hpp"
                tests_path.write_text(item["tests_code"])
                
                reference_path = checker_dir / "reference.cu"
                reference_path.write_text(item["reference_code"])
                
                os.system(f"cp /root/checker.cu {str(checker_dir)}")
                
                os.chdir(checker_dir)
                compile_result = os.system("make 2>&1")
                if compile_result != 0:
                    yield "data: " + json.dumps({
                        "status": "error",
                        "error": "Compilation failed",
                        "details": os.popen("make 2>&1").read(),
                        "test_results": [],
                        "passed_tests": 0,
                        "total_tests": 0
                    }) + "\n\n"
                    return
                
                yield "data: " + json.dumps({"status": "running"}) + "\n\n"
                
                # Stream the output line by line
                process = subprocess.Popen(["./checker"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                test_results = []
                passed_tests = 0
                total_tests = 0
                has_failed = False
                                
                # Process each line as it comes in the stream
                for line in iter(process.stdout.readline, ''):
                    if not line.strip():
                        continue

                    # This would indicate the end of the test results
                    if line.strip() in ["PASSED", "FAILED"]:
                        overall_status = line.strip()
                        continue
                    
                    test_case, name, status = line.split(',')   
                    test_id = int(test_case.split("/")[0])
                    total_tests = int(test_case.split("/")[1])

                    test_result = {
                        "test_id": test_id,
                        "name": name,
                        "status": status.strip()
                    }
                    test_results.append(test_result)
                    
                    if status.strip() == "PASSED":
                        passed_tests += 1
                        yield "data: " + json.dumps({
                            "status": "test_result",
                            "result": test_result,
                            "totalTests": total_tests
                        }) + "\n\n"
                    else:
                        has_failed = True
                        yield "data: " + json.dumps({
                            "status": "test_result",
                            "result": test_result,
                            "totalTests": total_tests
                        }) + "\n\n"
                
                stderr_output = process.stderr.read()

                if stderr_output:
                    yield "data: " + json.dumps({
                        "status": "error",
                        "error": "Runtime error",
                        "details": result.stderr,
                        "test_results": [],
                        "passed_tests": 0,
                        "total_tests": 0
                    }) + "\n\n"
                    return
                
                # Finally, at the very end, we can send the overall status
                yield "data: " + json.dumps({
                    "status": "complete",
                    "passed": not has_failed,
                    "test_results": test_results,
                    "passed_tests": passed_tests,
                    "total_tests": total_tests
                }) + "\n\n"

        except Exception as e:
            yield "data: " + json.dumps({
                "status": "error",
                "error": str(e),
                "test_results": []
            }) + "\n\n"

    return StreamingResponse(generate_checker_results(), media_type="text/event-stream")

async def generic_benchmark(item: dict):
    """Common implementation for all benchmark endpoints."""
    async def generate_benchmark_results():
        import subprocess
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                yield "data: " + json.dumps({"status": "compiling"}) + "\n\n"
                
                # Setup files
                solution_path = Path(tmpdir) / "solution.cu"
                solution_path.write_text(item["solution_code"])
                
                tests_path = Path(tmpdir) / "tests.hpp"
                tests_path.write_text(item["tests_code"])
                
                os.system("cp /root/benchmark.cu /root/core.hpp /root/Makefile " + tmpdir)            

                # Compile
                os.chdir(tmpdir)
                compile_result = os.system("make 2>&1")
                if compile_result != 0:
                    yield "data: " + json.dumps({
                        "status": "error",
                        "error": "Compilation failed", 
                        "details": os.popen("make 2>&1").read()
                    }) + "\n\n"
                    return
                
                yield "data: " + json.dumps({"status": "running"}) + "\n\n"
                
                # Run benchmark
                process = subprocess.Popen(["./benchmark"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                test_results = []
                total_tests = 0
                test_count = 0

                # Process each line as it comes in the stream
                for line in iter(process.stdout.readline, ''):
                    if not line.strip():
                        continue
                    
                    # Check if it's the last line with average GFLOPS
                    try:
                        avg_gflops = float(line.strip())
                        continue
                    except ValueError:
                        pass
                        
                    try:
                        test_id, name, runtime_ms, gflops = line.split(',')
                        test_result = {
                            "test_id": int(test_id),
                            "name": name,
                            "runtime_ms": float(runtime_ms),
                            "gflops": float(gflops)
                        }
                        test_results.append(test_result)
                        test_count += 1
                        
                        yield "data: " + json.dumps({
                            "status": "test_result",
                            "result": test_result,
                            "totalTests": test_count
                        }) + "\n\n"
                    except Exception as e:
                        yield "data: " + json.dumps({
                            "status": "error",
                            "error": "Failed to parse benchmark line",
                            "details": str(e),
                            "line": line
                        }) + "\n\n"
    
                stderr_output = process.stderr.read()
                
                if stderr_output:
                    yield "data: " + json.dumps({
                        "status": "error",
                        "error": "Runtime error",
                        "details": stderr_output
                    }) + "\n\n"
                    return

                # Finally, at the very end, we can send the overall status
                # Calculate average GFLOPS if not already calculated
                if not 'avg_gflops' in locals():
                    avg_gflops = sum(result["gflops"] for result in test_results) / len(test_results) if test_results else 0

                yield "data: " + json.dumps({
                    "status": "success",
                    "test_results": test_results,
                    "average_gflops": avg_gflops,
                    "total_tests": test_count
                }) + "\n\n"

        except Exception as e:
            yield "data: " + json.dumps({
                "status": "error",
                "error": str(e)
            }) + "\n\n"

    return StreamingResponse(generate_benchmark_results(), media_type="text/event-stream")

# GPU-specific endpoints
@app.function(gpu="T4")
@modal.web_endpoint(method="POST")
async def checker_t4(item: dict):
    return await generic_checker(item)

@app.function(gpu="H100")
@modal.web_endpoint(method="POST")
async def checker_h100(item: dict):
    return await generic_checker(item)

@app.function(gpu="A100-80GB")
@modal.web_endpoint(method="POST")
async def checker_a100_80gb(item: dict):
    return await generic_checker(item)

@app.function(gpu="A10G")
@modal.web_endpoint(method="POST")
async def checker_a10g(item: dict):
    return await generic_checker(item)

# GPU-specific endpoints
@app.function(gpu="T4")
@modal.web_endpoint(method="POST")
async def benchmark_t4(item: dict):
    return await generic_benchmark(item)

@app.function(gpu="H100")
@modal.web_endpoint(method="POST")
async def benchmark_h100(item: dict):
    return await generic_benchmark(item)

@app.function(gpu="A100-80GB")
@modal.web_endpoint(method="POST")
async def benchmark_a100_80gb(item: dict):
    return await generic_benchmark(item)

@app.function(gpu="A10G")
@modal.web_endpoint(method="POST")
async def benchmark_a10g(item: dict):
    return await generic_benchmark(item)