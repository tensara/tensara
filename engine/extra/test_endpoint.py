import requests

def test_benchmark():
    with open("solution.cu", "r") as f:
        solution_code = f.read()
    
    response = requests.post(
        "https://labs-asterisk--tensara-benchmark.modal.run",
        json={"code": solution_code} 
    )

    if response.status_code == 200:
        results = response.json()
        print(results)
    else:
        print("Error:", response.json())

def test_checker():
    with open("solution.cu", "r") as f:
        solution_cu = f.read()
    
    with open("cuda_bindings.cpp", "r") as f:
        cuda_bindings = f.read()
    
    with open("reference.py", "r") as f:
        reference_py = f.read()
    
    response = requests.post(
        "https://labs-asterisk--tensara-checker.modal.run",
        json={
            "solution_cu": solution_cu,
            "cuda_bindings": cuda_bindings,
            "reference_py": reference_py
        }
    )

    if response.status_code == 200:
        results = response.json()
        print("Checker Results:")
        print(results)
    else:
        print("Error:", response.json())

if __name__ == "__main__":
    test_benchmark()
    print("\nTesting Checker:")
    test_checker()


