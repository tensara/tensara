import modal
import tempfile
import os
from pathlib import Path

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
        "fastapi",
        "uvicorn", 
        "torch",
        "ninja"
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

@app.function(gpu="T4")
@modal.web_endpoint(method="POST")
def benchmark(item: dict):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            solution_path = Path(tmpdir) / "solution.cu"
            solution_path.write_text(item["solution_code"])
            
            tests_path = Path(tmpdir) / "tests.hpp"
            tests_path.write_text(item["tests_code"])
            
            os.system("cp /root/benchmark.cu /root/core.hpp /root/Makefile " + tmpdir)            

            os.chdir(tmpdir)
            compile_result = os.system("make 2>&1")
            if compile_result != 0:
                return {"error": "Compilation failed", "details": os.popen("make 2>&1").read()}
            
            import subprocess
            result = subprocess.run(["./benchmark"], capture_output=True, text=True)
            
            if result.stderr:
                return {"error": "Runtime error", "details": result.stderr}
            
            try:
                lines = result.stdout.strip().split('\n')
                test_results = []
                
                for line in lines[:-1]:
                    test_id, name, runtime_ms, gflops = line.split(',')
                    test_results.append({
                        "test_id": int(test_id),
                        "name": name,
                        "runtime_ms": float(runtime_ms),
                        "gflops": float(gflops)
                    })
                
                avg_gflops = float(lines[-1])
                
                return {
                    "status": "success",
                    "test_results": test_results,
                    "average_gflops": avg_gflops
                }
            except Exception as e:
                return {"error": "Failed to parse benchmark output", "details": str(e)}
            
    except Exception as e:
        return {"error": str(e)}


@app.function(gpu="T4")
@modal.web_endpoint(method="POST")
def checker(item: dict):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
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
                return {
                    "passed": False,
                    "error": "Compilation failed",
                    "details": os.popen("make 2>&1").read(),
                    "test_results": [],
                    "passed_tests": 0,
                    "total_tests": 0
                }
            
            import subprocess
            result = subprocess.run(["./checker"], capture_output=True, text=True)
            
            if result.stderr:
                return {
                    "passed": False,
                    "error": "Runtime error",
                    "details": result.stderr,
                    "test_results": [],
                    "passed_tests": 0, 
                    "total_tests": 0
                }
            
            lines = result.stdout.strip().split('\n')
            test_results = []
            passed_tests = 0
            
            for line in lines[:-1]:
                test_id, name, status = line.split(',')
                test_results.append({
                    "test_id": int(test_id),
                    "name": name,
                    "status": status.strip()
                })
                if status.strip() == "PASSED":
                    passed_tests += 1
            
            overall_status = lines[-1].strip()
            total_tests = len(test_results)
            
            return {
                "passed": overall_status == "PASSED",
                "test_results": test_results,
                "passed_tests": passed_tests,
                "total_tests": total_tests
            }
            
    except Exception as e:
        return {
            "passed": False,
            "error": str(e),
            "test_results": []
        }

