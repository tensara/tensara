import requests
import json

def test_benchmark():
    with open("solution.cu", "r") as f:
        solution_code = f.read()
    
    with open("tests.hpp", "r") as f:
        tests_code = f.read()
    
    response = requests.post(
        "https://labs-asterisk--tensara-benchmark.modal.run",
        json={
            "solution_code": solution_code,
            "tests_code": tests_code,
            "gpu_type": "T4"
        }
    )
    if response.status_code == 200:
        results = response.json()
        print("Benchmark Results:")
        print("Test Case\tTime (ms)\tGFLOPS")
        print("-" * 40)
        
        for test in results["test_results"]:
            print(f"{test['test_id']}\t\t{test['runtime_ms']:.3f}\t\t{test['gflops']:.3f}")
        
        print("-" * 40)
        print(f"Average GFLOPS: {results['average_gflops']:.3f}")
    else:
        print("Error:", response.json())

def test_checker():
    with open("solution.cu", "r") as f:
        solution_code = f.read()
    
    with open("tests.hpp", "r") as f:
        tests_code = f.read()

    with open("reference.cu", "r") as f:
        reference_code = f.read()

    response = requests.post(
        "https://labs-asterisk--tensara-checker.modal.run",
        json={
            "solution_code": solution_code,
            "tests_code": tests_code,
            "reference_code": reference_code,
            "gpu_type": "T4"
        }
    )

    if response.status_code == 200:
        results = response.json()
        print("\nChecker Results:")
        print("Test Case\tStatus")
        print("-" * 30)
        
        for test in results["test_results"]:
            print(f"{test['test_id']}\t\t{test['status']}")
        
        print("-" * 30)
        print(f"Overall: {'PASSED' if results['passed'] else 'FAILED'}")
        
        if not results['passed'] and 'error' in results:
            print(f"\nError: {results['error']}")
            if 'details' in results:
                print(f"Details: {results['details']}")
    else:
        print("Error:", response.json())

if __name__ == "__main__":
    test_benchmark()
    # test_checker()


