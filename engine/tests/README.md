# Testing the engine for a particular problem

There is a script called "submit.sh" that can be used to test the engine for a particular problem.

**IMPORTANT**: You have to put the solution in the solution.cu file in the tests directory. Make sure the kernel is launched from a mehtod called "solution" in the solution.cu file.

Example:

```cpp
__global__ void solution(int* input, int* output) {
    // Your solution code here
}
```

This script will automatically make a request.json and send it to the modal endpoint.

Usage:

```bash
./submit.sh <problem_name> <modal_endpoint>
```

Example:

```bash
./submit.sh "test" "http://localhost:8000"
```

This will submit the problem "test" to the modal endpoint "http://localhost:8000" and show the results in the terminal.