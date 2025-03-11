#!/bin/bash

if [ $# -lt 1 ]; then
  echo "Usage: $0 <problem_name> [api_endpoint]"
  echo "Example: $0 matrix_multiplication https://api.example.com/submit"
  exit 1
fi

problem_name="$1"
api_endpoint="${2:-https://api.example.com/submit}"

if [ ! -f "solution.cu" ]; then
  echo "Error: solution.cu file not found in the current directory."
  exit 1
fi

solution_code=$(cat solution.cu)
        
echo "Creating request.json..."
cat > request.json << EOF
{
  "solution_code": $(printf '%s' "$solution_code" | jq -Rs .),
  "problem": "$problem_name",
  "gpu": "T4"
}
EOF

echo "Created request.json successfully."

echo "Sending POST request to $api_endpoint..."
curl -s -X POST --no-buffer \
  -H "Content-Type: application/json" \
  -d @request.json \
  "$api_endpoint"

echo "Done."