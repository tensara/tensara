#!/bin/bash

mode="cuda"  # default mode
endpoint_type="benchmark_cli"  # default endpoint
if [[ "$1" == "--python" ]]; then
  mode="python"
  shift
elif [[ "$1" == "--cuda" ]]; then
  mode="cuda"
  shift
elif [[ "$1" == "--mojo" ]]; then
  mode="mojo"
  shift
elif [[ "$1" == "--cute" ]]; then
  mode="cute"
  shift
elif [[ "$1" == "--sandbox" ]]; then
  endpoint_type="sandbox"
  mode="cuda"  # default for sandbox
  shift
  # Check if next argument is a language flag for sandbox
  if [[ "$1" == "--cuda" ]]; then
    mode="cuda"
    shift
  elif [[ "$1" == "--cute" ]]; then
    mode="cute"
    shift
  fi
fi

if [[ "$endpoint_type" == "sandbox" ]]; then
  if [ $# -lt 0 ]; then
    echo "Usage: $0 --sandbox [--cuda|--cute] [api_endpoint]"
    echo "Example: $0 --sandbox --cuda https://api.example.com/sandbox-T4"
    echo "Example: $0 --sandbox --cute https://api.example.com/sandbox-T4"
    exit 1
  fi
  api_endpoint="${1:-https://api.example.com/sandbox-T4}"
else
  if [ $# -lt 1 ]; then
    echo "Usage: $0 [--python|--cuda|--mojo|--cute|--sandbox] <problem_name> [api_endpoint]"
    echo "Example: $0 --cuda matrix_multiplication https://api.example.com/benchmark_cli-T4"
    echo "Example: $0 --sandbox --cute https://api.example.com/sandbox-T4"
    exit 1
  fi
  problem_name="$1"
  api_endpoint="${2:-https://api.example.com/benchmark_cli-T4}"
fi

case $mode in
  "python")
    solution_file="solution.py"
    language="python"
    ;;
  "cuda")
    solution_file="solution.cu"
    language="cuda"
    ;;
  "mojo")
    solution_file="solution.mojo"
    language="mojo"
    ;;
  "cute")
    solution_file="solution.py"
    language="cute"
    ;;
esac

if [ ! -f "$solution_file" ]; then
  echo "Error: $solution_file file not found in the current directory."
  exit 1
fi

solution_code=$(cat "$solution_file")

if [[ "$endpoint_type" == "sandbox" ]]; then
  echo "Creating request.json for sandbox..."
  cat > request.json << EOF
{
  "code": $(printf '%s' "$solution_code" | jq -Rs .),
  "language": "$language"
}
EOF
else
  if [ ! -f "problem.py" ]; then
    echo "Error: problem.py file not found in the current directory."
    exit 1
  fi
  
  problem_def=$(cat problem.py)
        
  echo "Creating request.json..."
  cat > request.json << EOF
{
  "code": $(printf '%s' "$solution_code" | jq -Rs .),
  "problem": "$problem_name",
  "problem_def": $(printf '%s' "$problem_def" | jq -Rs .),
  "gpu": "T4",
  "dtype": "float32",
  "language": "$language"
}
EOF
fi

echo "Created request.json successfully."

echo "Sending POST request to $api_endpoint..."
curl -s -X POST --no-buffer \
  -H "Content-Type: application/json" \
  -d @request.json \
  "$api_endpoint"

echo "Done."