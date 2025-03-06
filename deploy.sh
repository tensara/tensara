#!/bin/bash

# Define paths
APP_DIR="/home/ubuntu/code/tensara"
BUILD_DIR="${APP_DIR}/.next-build-temp"
CURRENT_BUILD_DIR="${APP_DIR}/.next"

# Load environment variables to ensure pnpm is found
export NVM_DIR="$HOME/.nvm"
source "$NVM_DIR/nvm.sh"
source ~/.profile
source ~/.bashrc

# Check if pnpm is installed
if ! command -v pnpm &> /dev/null; then
    echo "❌ Error: pnpm not found. Ensure pnpm is installed and in the PATH."
    exit 1
fi

# Pull latest code
cd "$APP_DIR" || { echo "❌ Error: Failed to change directory to $APP_DIR"; exit 1; }
git pull origin main || { echo "❌ Error: Git pull failed"; exit 1; }

# Install dependencies
pnpm install || { echo "❌ Error: Dependency installation failed"; exit 1; }

# Build Next.js app
pnpm build || { echo "❌ Error: Build failed"; exit 1; }

# Ensure build was successful before proceeding
if [ -d "$BUILD_DIR" ]; then
    # Swap new build atomically
    mv "$CURRENT_BUILD_DIR" "${CURRENT_BUILD_DIR}-old"
    mv "$BUILD_DIR" "$CURRENT_BUILD_DIR"

    # Reload Next.js server with PM2 (zero downtime)
    pm2 reload tensara || { echo "❌ Error: PM2 reload failed"; exit 1; }

    # Cleanup
    rm -rf "${CURRENT_BUILD_DIR}-old"
    echo "✅ Deployment successful!"
    exit 0  # Explicit success exit
else
    echo "❌ Build directory not found. Deployment aborted."
    exit 1
fi