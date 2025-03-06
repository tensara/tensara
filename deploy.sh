#!/bin/bash

# Define paths
APP_DIR="/home/ubuntu/code/tensara"
BUILD_DIR="$APP_DIR/.next-build-temp"

# Pull latest code
cd $APP_DIR
git pull origin main

# Install dependencies
pnpm install

# Build in a separate directory
pnpm build --output $BUILD_DIR

# Ensure build was successful before proceeding
if [ -d "$BUILD_DIR" ]; then
    # Swap new build atomically
    mv $APP_DIR/.next $APP_DIR/.next-old
    mv $BUILD_DIR $APP_DIR/.next

    # Reload Next.js server with PM2 (zero downtime)
    pm2 reload tensara

    # Cleanup
    rm -rf $APP_DIR/.next-old
    echo "✅ Deployment successful!"
else
    echo "❌ Build failed. Deployment aborted."
fi
