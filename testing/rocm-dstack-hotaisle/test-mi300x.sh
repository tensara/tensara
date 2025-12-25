#!/bin/bash
# Test script for AMD MI300X provisioning via dstack
# Usage: ./test-mi300x.sh [test_type]
# test_type: cold (default), warm, or cleanup

set -e

TEST_DIR="/Users/somesh/projects/stk/tensara/tensara-app/testing/rocm-dstack-hotaisle"
TEST_TYPE="${1:-cold}"

cd "$TEST_DIR"

echo "==================================================="
echo "AMD MI300X Test Script"
echo "==================================================="
echo "Test Type: $TEST_TYPE"
echo "Directory: $TEST_DIR"
echo "Time: $(date)"
echo ""

case "$TEST_TYPE" in
  cold)
    echo "Running COLD START test..."
    echo "This will provision a new MI300X VM (2-5 minutes)"
    echo ""
    echo "Press Ctrl+C to cancel, or wait 5 seconds to continue..."
    sleep 5
    
    echo ""
    echo "Cleaning up any existing VMs..."
    dstack stop -y || true
    
    echo ""
    echo "Starting fresh MI300X provision..."
    time dstack apply -f .dstack.yml -y
    ;;
    
  warm)
    echo "Running WARM REUSE test..."
    echo "This should reuse existing MI300X VM (< 10 seconds)"
    echo "Make sure you ran a cold start within the last 10 minutes!"
    echo ""
    echo "Press Ctrl+C to cancel, or wait 3 seconds to continue..."
    sleep 3
    
    echo ""
    echo "Checking for warm VMs..."
    dstack ps
    
    echo ""
    echo "Starting warm reuse test..."
    time dstack apply -f .dstack.yml -y
    ;;
    
  cleanup)
    echo "Cleaning up all VMs..."
    echo ""
    dstack ps
    echo ""
    echo "Stopping all runs..."
    dstack stop -y || echo "No runs to stop"
    echo ""
    echo "Cleanup complete!"
    ;;
    
  *)
    echo "ERROR: Unknown test type '$TEST_TYPE'"
    echo "Usage: $0 [cold|warm|cleanup]"
    exit 1
    ;;
esac

echo ""
echo "==================================================="
echo "Test complete!"
echo "==================================================="
