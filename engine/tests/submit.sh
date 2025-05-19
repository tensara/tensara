#!/bin/bash

mode="cuda"  # default mode
if [[ "$1" == "--triton" ]]; then
  mode="triton"
  shift
elif [[ "$1" == "--cuda" ]]; then
  mode="cuda"
  shift
elif [[ "$1" == "--mojo" ]]; then
  mode="mojo"
  shift
fi

if [ $# -lt 1 ]; then
  echo "Usage: $0 [--triton|--cuda|--mojo] <problem_name> [api_endpoint]"
  echo "Example: $0 --cuda matrix_multiplication https://api.example.com/submit"
  exit 1
fi

problem_name="$1"
api_endpoint="${2:-https://api.example.com/submit}"

case $mode in
  "triton")
    solution_file="solution.py"
    language="triton"
    ;;
  "cuda")
    solution_file="solution.cu"
    language="cuda"
    ;;
  "mojo")
    solution_file="solution.mojo"
    language="mojo"
    ;;
esac

if [ ! -f "$solution_file" ]; then
  echo "Error: $solution_file file not found in the current directory."
  exit 1
fi

solution_code=$(cat "$solution_file")
problem_def=$(cat problem.py)
        
echo "Creating request.json..."
cat > request.json << EOF
{
  "solution_code": $(printf '%s' "$solution_code" | jq -Rs .),
  "problem": "$problem_name",
  "problem_def": $(printf '%s' "$problem_def" | jq -Rs .),
  "gpu": "T4",
  "dtype": "float32",
  "language": "$language"
}
EOF

echo "Created request.json successfully."

echo "Sending POST request to $api_endpoint..."
curl -s -X POST --no-buffer \
  -H "Content-Type: application/json" \
  -d @request.json \
  "$api_endpoint"

echo "Done."