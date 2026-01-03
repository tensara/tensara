#!/usr/bin/env python3
"""
Standalone Fleet Creation Script for AMD DevCloud

This script creates the AMD MI300X Backend Fleet required for task submission.
It should be run once before submitting tasks, or whenever the fleet needs to be recreated.

Usage:
    # Dry run (test configuration without creating fleet)
    export AMD_FLEET_DRY_RUN=true
    python3 create_amd_fleet.py

    # Create actual fleet
    export AMD_FLEET_DRY_RUN=false
    python3 create_amd_fleet.py

    # Verify fleet was created
    dstack fleet list
"""

import os
import sys
import logging
from pathlib import Path

# Add engine directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from amd_fleet_manager import AMDFleetManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Create AMD DevCloud fleet"""
    
    print("=" * 70)
    print("AMD DevCloud Backend Fleet Creation")
    print("=" * 70)
    print()
    
    # Initialize fleet manager
    logger.info("Initializing AMD Fleet Manager...")
    manager = AMDFleetManager()
    
    # Display configuration
    print("Configuration:")
    print(f"  Backend:       {manager.amd_config['backend']}")
    print(f"  Fleet Name:    {manager.amd_config['fleet_name']}")
    print(f"  GPU Type:      {manager.amd_config['gpu_type']}")
    print(f"  GPU Memory:    {manager.amd_config['gpu_memory']}")
    print(f"  Max Nodes:     {manager.amd_config['max_nodes']}")
    print(f"  Idle Duration: {manager.amd_config['idle_duration']}")
    print(f"  Spot Policy:   {manager.amd_config['spot_policy']}")
    print(f"  Dry Run:       {manager.amd_config.get('dry_run', False)}")
    print()
    
    # Check if dry run
    if manager.amd_config.get('dry_run', False):
        print("🧪 DRY RUN MODE ENABLED")
        print("Fleet will NOT be actually created. This is a test run.")
        print()
    
    # Ask for confirmation (unless dry run)
    if not manager.amd_config.get('dry_run', False):
        print("⚠️  This will create a fleet that may incur costs ($1.99/hour when active).")
        print("The fleet will auto-scale from 0 to 1 instance on-demand.")
        print()
        response = input("Continue with fleet creation? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("Aborted.")
            return 1
        print()
    
    # Create fleet
    fleet_name = manager.amd_config['fleet_name']
    logger.info(f"Creating fleet '{fleet_name}'...")
    
    success = manager.ensure_fleet_exists(fleet_name)
    
    print()
    print("=" * 70)
    if success:
        if manager.amd_config.get('dry_run', False):
            print("✅ DRY RUN SUCCESSFUL")
            print()
            print("Configuration is valid. Set AMD_FLEET_DRY_RUN=false to create actual fleet.")
        else:
            print("✅ FLEET CREATED SUCCESSFULLY")
            print()
            print(f"Fleet '{fleet_name}' is ready for task submission.")
            print()
            print("Verify with: dstack fleet list")
            print("View details: dstack fleet list --verbose")
        print("=" * 70)
        return 0
    else:
        print("❌ FLEET CREATION FAILED")
        print()
        print("Please check the error messages above.")
        print()
        print("Common issues:")
        print("  1. dstack not authenticated (run: dstack config)")
        print("  2. AMD DevCloud backend not configured")
        print("  3. Network connectivity issues")
        print("  4. Invalid configuration values")
        print()
        print("For help, see: FLEET_API_MIGRATION.md")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
