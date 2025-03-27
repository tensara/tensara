#!/bin/bash

python_mode=false
if [[ "$1" == "--python" ]]; then
  python_mode=true
  shift
fi

if [ $# -lt 1 ]; then
  echo "Usage: $0 [--python] <problem_name> [api_endpoint]"
  echo "Example: $0 matrix_multiplication https://api.example.com/submit"
  exit 1
fi

problem_name="$1"
api_endpoint="${2:-https://api.example.com/submit}"


if $python_mode; then
  solution_file="solution.py"
  language="python"
else
  solution_file="solution.cu"
  language="cuda"
fi

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