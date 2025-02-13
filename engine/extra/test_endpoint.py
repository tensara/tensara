import requests

def test_benchmark():
    with open("example_solution.cuh", "r") as f:
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

if __name__ == "__main__":
    test_benchmark()
