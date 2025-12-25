#!/bin/bash
# AMD DevCloud Setup Script for Tensara
# 
# This script helps configure AMD DevCloud access for dstack fleet management.
# Run this script to set up the necessary environment variables and SSH keys.

set -e

echo "🚀 AMD DevCloud Setup for Tensara"
echo "================================="
echo

# Check if running in correct directory
if [[ ! -f "engine/amd_fleet_manager.py" ]]; then
    echo "❌ Error: Please run this script from the tensara-app root directory"
    echo "   Current directory: $(pwd)"
    echo "   Expected files: engine/amd_fleet_manager.py"
    exit 1
fi

# Create .env file for AMD DevCloud configuration
ENV_FILE=".env.amd"
echo "📝 Creating AMD DevCloud configuration file: $ENV_FILE"

cat > "$ENV_FILE" << 'EOF'
# AMD DevCloud Configuration for Tensara
# 
# This file contains environment variables needed for AMD DevCloud fleet management.
# Copy these variables to your main .env file or source this file directly.

# AMD DevCloud SSH Configuration
# Replace these with your actual AMD DevCloud credentials
AMD_DEVCLOUD_SSH_USER=your_amd_username
AMD_DEVCLOUD_SSH_KEY_PATH=~/.ssh/amd_devcloud_rsa
AMD_DEVCLOUD_HOSTS=amd-host1.example.com,amd-host2.example.com

# Fleet Configuration
AMD_FLEET_NAME=amd-mi300x-fleet
AMD_DEVCLOUD_DISCOVERY=true

# VM Configuration (optional overrides)
AMD_VM_IDLE_TIMEOUT=600
AMD_VM_MAX_CONCURRENT=3
AMD_MI300X_HOURLY_RATE=1.99

# Grant Tracking (optional)
AMD_GRANT_CREDITS_TOTAL=1400.00
AMD_COST_ALERT_THRESHOLD=0.8
EOF

echo "✅ Created $ENV_FILE"
echo

# Check for SSH key
SSH_KEY_PATH="$HOME/.ssh/amd_devcloud_rsa"
echo "🔑 Checking SSH key configuration..."

if [[ ! -f "$SSH_KEY_PATH" ]]; then
    echo "⚠️  SSH key not found at: $SSH_KEY_PATH"
    echo
    echo "To set up SSH key for AMD DevCloud:"
    echo "1. Generate SSH key pair:"
    echo "   ssh-keygen -t rsa -b 4096 -f ~/.ssh/amd_devcloud_rsa -C 'tensara-amd-devcloud'"
    echo
    echo "2. Add public key to AMD DevCloud:"
    echo "   cat ~/.ssh/amd_devcloud_rsa.pub"
    echo "   (Copy this to your AMD DevCloud SSH keys)"
    echo
    echo "3. Update $ENV_FILE with your actual AMD DevCloud details"
    echo
else
    echo "✅ SSH key found at: $SSH_KEY_PATH"
    
    # Check key permissions
    KEY_PERMS=$(stat -c %a "$SSH_KEY_PATH" 2>/dev/null || stat -f %A "$SSH_KEY_PATH" 2>/dev/null || echo "unknown")
    if [[ "$KEY_PERMS" != "600" ]]; then
        echo "🔧 Fixing SSH key permissions..."
        chmod 600 "$SSH_KEY_PATH"
        echo "✅ SSH key permissions set to 600"
    else
        echo "✅ SSH key permissions are correct (600)"
    fi
fi

# Check dstack installation
echo "🔍 Checking dstack installation..."
if command -v dstack >/dev/null 2>&1; then
    DSTACK_VERSION=$(dstack --version 2>/dev/null || echo "unknown")
    echo "✅ dstack CLI found: $DSTACK_VERSION"
    
    # Check dstack project configuration
    echo "📋 Checking dstack project configuration..."
    if dstack project list >/dev/null 2>&1; then
        echo "✅ dstack project configured"
        
        # Show current project
        echo "Current dstack projects:"
        dstack project list | head -5
    else
        echo "⚠️  dstack project not configured"
        echo "Run: dstack project add <project-name> <server-url>"
    fi
else
    echo "❌ dstack CLI not found"
    echo "Install dstack: pip install dstack"
    exit 1
fi

# Test fleet management
echo "🧪 Testing fleet manager..."
if python3 -c "
import sys
sys.path.insert(0, 'engine')
try:
    from amd_fleet_manager import AMDFleetManager
    manager = AMDFleetManager()
    print('✅ Fleet manager imports successfully')
    print(f'🏗️  Fleet name: {manager.amd_config[\"fleet_name\"]}')
    print(f'👤 SSH user: {manager.amd_config[\"ssh_user\"]}')
    print(f'🔑 SSH key: {manager.amd_config[\"ssh_key_path\"]}')
    print(f'🖥️  Hosts: {len(manager.amd_config[\"hosts\"])} configured')
except Exception as e:
    print(f'❌ Fleet manager error: {e}')
    sys.exit(1)
" 2>/dev/null; then
    echo
else
    echo "❌ Fleet manager test failed"
    echo "Check Python dependencies: pyyaml"
    exit 1
fi

# Instructions
echo "📋 Next Steps:"
echo "============="
echo
echo "1. 📝 Edit the configuration file:"
echo "   nano $ENV_FILE"
echo
echo "2. 🔑 Set up your AMD DevCloud SSH access:"
echo "   - Replace 'your_amd_username' with your actual username"
echo "   - Replace host URLs with actual AMD DevCloud endpoints"
echo "   - Ensure SSH key is properly configured"
echo
echo "3. 🚀 Load the configuration:"
echo "   source $ENV_FILE"
echo
echo "4. 🧪 Test fleet creation:"
echo "   python3 engine/amd_fleet_manager.py ensure"
echo
echo "5. ✅ Verify setup:"
echo "   dstack fleet list"
echo
echo "📚 For troubleshooting, check the documentation or run:"
echo "   python3 engine/amd_fleet_manager.py status"
echo

echo "🎉 AMD DevCloud setup completed!"
echo "Don't forget to update $ENV_FILE with your actual credentials."