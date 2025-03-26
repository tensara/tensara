import tempfile
import subprocess
from pathlib import Path


def run_checker(binary: bytes):
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(binary)
        f.close()

    path = Path(f.name)
    path.chmod(0o755)

    yield {"status": "running"}

    checker = subprocess.Popen(
        [str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    test_results = []
    has_failed = False
    passed_tests = 0

    for line in iter(checker.stdout.readline, ""):
        line = line.strip()
        if not line:
            continue

        # indicates the end of the test results
        if line in ["PASSED", "FAILED"]:
            continue

        test_case, name, status = line.split(",")
        status = status.strip()
        test_id = int(test_case.split("/")[0])
        total_tests = int(test_case.split("/")[1])

        test_result = {"test_id": test_id, "name": name, "status": status}
        test_results.append(test_result)
        yield {"status": "test_result", "result": test_result, "totalTests": total_tests}

        if status == "PASSED":
            passed_tests += 1
        else:
            has_failed = True

    try:
        checker.wait(timeout=1)
    except subprocess.TimeoutExpired:
        checker.kill()

    stderr = checker.stderr.read()
    path.unlink()

    if stderr:
        yield {
            "status": "error",
            "error": "Runtime error",
            "details": stderr,
            "test_results": [],
            "passed_tests": 0,
            "total_tests": 0,
        }

    yield {
        "status": "complete",
        "passed": not has_failed,
        "test_results": test_results,
        "passed_tests": passed_tests,
        "total_tests": total_tests,
    }


def run_benchmark(binary: bytes):
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(binary)
        f.close()

    path = Path(f.name)
    path.chmod(0o755)

    yield {"status": "compiling"}
    yield {"status": "running"}
    benchmark = subprocess.Popen(
        [str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    test_results = []
    test_count = 0
    avg_gflops = None

    for line in iter(benchmark.stdout.readline, ""):
        line = line.strip()
        if not line:
            continue

        try:
            avg_gflops = float(line)
            continue
        except ValueError:
            pass

        try:
            test_id, name, runtime_ms, gflops = line.split(",")
            test_result = {
                "test_id": int(test_id),
                "name": name,
                "runtime_ms": float(runtime_ms),
                "gflops": float(gflops),
            }
            test_results.append(test_result)
            test_count += 1
            yield {"status": "test_result", "result": test_result, "totalTests": test_count}
        except Exception as e:
            yield {
                "status": "error",
                "error": "Failed to parse benchmark line",
                "details": str(e),
                "line": line,
            }

    try:
        benchmark.wait(timeout=1)
    except subprocess.TimeoutExpired:
        benchmark.kill()

    stderr = benchmark.stderr.read()
    path.unlink()

    if stderr:
        yield {
            "status": "error",
            "error": "Runtime error",
            "details": stderr,
        }
        return

    # Finally, at the very end, we can send the overall status
    # Calculate average GFLOPS if not already calculated
    if not avg_gflops:
        avg_gflops = (
            sum(result["gflops"] for result in test_results) / len(test_results)
            if len(test_results)
            else 0
        )

    yield {
        "status": "success",
        "test_results": test_results,
        "average_gflops": avg_gflops,
        "total_tests": test_count,
    }
