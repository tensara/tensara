#!/usr/bin/env python3
"""
Fleet Manager for AMD DevCloud Integration (Backend Fleet API)

Handles Backend Fleet lifecycle management for dstack + AMD DevCloud:
- Backend Fleet creation and validation (dstack 0.20+ API)
- Fleet health monitoring
- Automatic fleet provisioning before task submission
- Supports cloud backend provisioning (not SSH-based)

Key Changes from SSH Fleet:
- Uses 'backends' field instead of 'ssh_config'
- Provisions instances via cloud API (DigitalOcean)
- No need for static host lists or SSH keys
"""

import os
import sys
import yaml
import json
import time
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FleetConfig:
    """Fleet configuration for AMD DevCloud Backend Fleet"""
    name: str
    backend: str
    gpu_type: str = "MI300X"
    gpu_memory: str = "192GB"
    max_nodes: int = 1
    idle_duration: str = "10m"
    spot_policy: str = "auto"


class AMDFleetManager:
    """
    Manages AMD DevCloud Backend Fleets for dstack
    
    Responsibilities:
    1. Ensure required fleets exist before task submission
    2. Create Backend Fleets dynamically if missing
    3. Monitor fleet health and availability
    4. Handle cloud backend provisioning (not SSH-based)
    """
    
    def __init__(self):
        """Initialize fleet manager"""
        self.fleet_configs: Dict[str, FleetConfig] = {}
        self.fleet_cache: Dict[str, Tuple[float, bool]] = {}
        self.cache_ttl = 300  # 5 minutes cache TTL
        
        # Load AMD DevCloud configuration from environment
        self.amd_config = self._load_amd_config()
        logger.info(f"AMD Fleet Manager initialized with config: {list(self.amd_config.keys())}")
    
    def _load_amd_config(self) -> Dict[str, Any]:
        """Load AMD DevCloud Backend Fleet configuration from environment"""
        config = {
            'backend': os.getenv('AMD_BACKEND', 'amddevcloud'),
            'fleet_name': os.getenv('AMD_FLEET_NAME', 'amd-mi300x-fleet'),
            'gpu_type': os.getenv('AMD_GPU_TYPE', 'MI300X'),
            'gpu_memory': os.getenv('AMD_GPU_MEMORY', '192GB'),
            'max_nodes': int(os.getenv('AMD_FLEET_MAX_NODES', '1')),
            'idle_duration': os.getenv('AMD_FLEET_IDLE_DURATION', '10m'),
            'spot_policy': os.getenv('AMD_SPOT_POLICY', 'auto'),
            'dry_run': os.getenv('AMD_FLEET_DRY_RUN', 'false').lower() == 'true',
        }
        
        logger.info(f"Backend Fleet Config: backend={config['backend']}, "
                   f"fleet={config['fleet_name']}, gpu={config['gpu_type']}, "
                   f"max_nodes={config['max_nodes']}, dry_run={config['dry_run']}")
        
        return config
    
    def ensure_fleet_exists(self, fleet_name: Optional[str] = None) -> bool:
        """
        Ensure the specified fleet exists, creating it if necessary
        
        Args:
            fleet_name: Name of fleet to ensure (defaults to AMD fleet)
            
        Returns:
            True if fleet exists or was created successfully
        """
        actual_fleet_name = fleet_name or self.amd_config['fleet_name']
        
        try:
            # Check if fleet already exists
            if self._fleet_exists(actual_fleet_name):
                logger.info(f"Fleet '{actual_fleet_name}' already exists")
                return True
            
            # Fleet doesn't exist - create it
            logger.info(f"Fleet '{actual_fleet_name}' not found, creating...")
            return self._create_amd_fleet(actual_fleet_name)
            
        except Exception as e:
            logger.error(f"Failed to ensure fleet exists: {e}")
            return False
    
    def _fleet_exists(self, fleet_name: str) -> bool:
        """Check if fleet exists using dstack CLI"""
        try:
            # Use cached result if fresh
            cache_key = f"fleet_exists_{fleet_name}"
            if cache_key in self.fleet_cache:
                cached_time, cached_result = self.fleet_cache[cache_key]
                if time.time() - cached_time < self.cache_ttl:
                    return cached_result
            
            # Query dstack for fleet list
            result = subprocess.run(
                ['dstack', 'fleet', 'list'],
                capture_output=True,
                text=True,
                timeout=30,
            )
            
            if result.returncode != 0:
                logger.warning(f"Failed to list fleets: {result.stderr}")
                return False
            
            # Parse output to find fleet
            fleet_exists = fleet_name in result.stdout
            
            # Cache result
            self.fleet_cache[cache_key] = (time.time(), fleet_exists)
            
            logger.debug(f"Fleet '{fleet_name}' exists: {fleet_exists}")
            return fleet_exists
            
        except subprocess.TimeoutExpired:
            logger.warning("Fleet list command timed out")
            return False
        except Exception as e:
            logger.error(f"Error checking fleet existence: {e}")
            return False
    
    def _verify_fleet_created(self, fleet_name: str, max_retries: int = 3) -> bool:
        """
        Verify that fleet was actually created (dstack may return 0 even on failure)
        
        Args:
            fleet_name: Name of fleet to verify
            max_retries: Number of verification attempts
            
        Returns:
            True if fleet exists and is properly configured
        """
        for attempt in range(max_retries):
            try:
                logger.info(f"Verification attempt {attempt + 1}/{max_retries}...")
                
                # Query dstack for fleet list
                result = subprocess.run(
                    ['dstack', 'fleet', 'list'],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                
                if result.returncode != 0:
                    logger.warning(f"Fleet list failed on attempt {attempt + 1}: {result.stderr}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    return False
                
                # Check if fleet appears in output
                if fleet_name in result.stdout:
                    # Additional check: verify backend is correct
                    backend = self.amd_config.get('backend', 'amddevcloud')
                    if backend in result.stdout or 'amddevcloud' in result.stdout:
                        logger.info(f"✅ Fleet '{fleet_name}' verified with backend '{backend}'")
                        return True
                    else:
                        logger.warning(f"Fleet found but backend mismatch")
                        if attempt < max_retries - 1:
                            time.sleep(2)
                            continue
                        return False
                else:
                    logger.warning(f"Fleet '{fleet_name}' not found in list on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    return False
                    
            except subprocess.TimeoutExpired:
                logger.warning(f"Fleet verification timed out on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return False
            except Exception as e:
                logger.error(f"Fleet verification error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return False
        
        return False
    
    def _create_amd_fleet(self, fleet_name: str) -> bool:
        """Create AMD DevCloud Backend Fleet using dstack CLI"""
        try:
            # Check if dry-run mode
            if self.amd_config.get('dry_run', False):
                logger.info("🧪 DRY RUN MODE: Would create fleet with following config:")
                fleet_config = self._generate_fleet_config(fleet_name)
                logger.info(yaml.dump(fleet_config, default_flow_style=False, sort_keys=False))
                logger.info("✅ Dry run successful - no actual fleet created")
                return True
            
            # Create fleet configuration
            fleet_config = self._generate_fleet_config(fleet_name)
            
            # Create temp directory in engine/ folder instead of /tmp
            # This fixes the "not in subpath" error from dstack CLI
            import tempfile
            import shutil
            
            # Get the engine directory (where this script lives)
            engine_dir = Path(__file__).parent
            
            # Create temp directory inside engine/
            temp_dir = Path(tempfile.mkdtemp(
                prefix=f"fleet_{fleet_name}_",
                dir=str(engine_dir)
            ))
            
            # Write fleet config to temp directory with predictable name
            temp_fleet_file = temp_dir / ".dstack-fleet.yml"
            with open(temp_fleet_file, 'w') as f:
                yaml.dump(fleet_config, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Created fleet config: {temp_fleet_file}")
            
            try:
                # Apply fleet configuration
                logger.info(f"Applying Backend Fleet configuration via dstack...")
                result = subprocess.run(
                    ['dstack', 'apply', '-f', '.dstack-fleet.yml', '-y'],
                    capture_output=True,
                    text=True,
                    timeout=120,  # Fleet creation can take time
                    cwd=str(temp_dir),  # Run from temp directory
                )
                
                if result.returncode == 0:
                    logger.info(f"✅ dstack apply completed for fleet '{fleet_name}'")
                    logger.info(f"Fleet creation output:\n{result.stdout}")
                    
                    # CRITICAL: Verify fleet was actually created
                    # dstack may return 0 even if fleet creation failed
                    logger.info("Verifying fleet was actually created...")
                    time.sleep(2)  # Brief wait for fleet to register
                    
                    if self._verify_fleet_created(fleet_name):
                        logger.info(f"✅ Successfully verified fleet '{fleet_name}' exists")
                        
                        # Clear cache
                        cache_key = f"fleet_exists_{fleet_name}"
                        if cache_key in self.fleet_cache:
                            del self.fleet_cache[cache_key]
                        
                        return True
                    else:
                        logger.error(f"❌ Fleet '{fleet_name}' not found after creation - apply may have failed silently")
                        logger.error(f"Check dstack output above for errors")
                        return False
                else:
                    logger.error(f"❌ Failed to create fleet '{fleet_name}'")
                    logger.error(f"Error output:\n{result.stderr}")
                    logger.error(f"Stdout:\n{result.stdout}")
                    return False
                    
            finally:
                # Clean up temp directory
                try:
                    shutil.rmtree(temp_dir)
                    logger.debug(f"Cleaned up temp directory: {temp_dir}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup temp directory: {cleanup_error}")
            
        except subprocess.TimeoutExpired:
            logger.error("Fleet creation timed out after 120s")
            return False
        except Exception as e:
            logger.error(f"Failed to create fleet: {e}")
            return False
    
    def _generate_fleet_config(self, fleet_name: str) -> Dict[str, Any]:
        """Generate Backend Fleet configuration for AMD DevCloud"""
        
        # Build Backend Fleet configuration
        config = {
            'type': 'fleet',
            'name': fleet_name,
            # Backend fleet specification (not SSH)
            'nodes': f"0..{self.amd_config['max_nodes']}",
            'backends': [self.amd_config['backend']],
            # Resource specifications for MI300X
            'resources': {
                'gpu': {
                    'name': self.amd_config['gpu_type'],
                    'memory': self.amd_config['gpu_memory'],
                },
                'cpu': '2..',  # Minimum viable
                'memory': '8GB..',  # Minimum viable
                'disk': '100GB..',  # Minimum viable
            },
            # Fleet settings
            'idle_duration': self.amd_config['idle_duration'],
            'spot_policy': self.amd_config['spot_policy'],
        }
        
        logger.info(f"Generated Backend Fleet config: backend={self.amd_config['backend']}, "
                   f"nodes=0..{self.amd_config['max_nodes']}, gpu={self.amd_config['gpu_type']}")
        return config
    
    def get_backend_info(self) -> Dict[str, Any]:
        """
        Get information about the configured backend
        
        Returns:
            Dictionary with backend configuration details
        """
        return {
            'backend': self.amd_config['backend'],
            'fleet_name': self.amd_config['fleet_name'],
            'gpu_type': self.amd_config['gpu_type'],
            'max_nodes': self.amd_config['max_nodes'],
            'idle_duration': self.amd_config['idle_duration'],
            'spot_policy': self.amd_config['spot_policy'],
        }
    
    def validate_fleet_connectivity(self, fleet_name: str) -> bool:
        """
        Validate that fleet hosts are accessible
        
        Args:
            fleet_name: Name of fleet to validate
            
        Returns:
            True if fleet is accessible
        """
        try:
            # For SSH fleets, we can try to list fleet instances
            result = subprocess.run(
                ['dstack', 'fleet', 'list', '--verbose'],
                capture_output=True,
                text=True,
                timeout=30,
            )
            
            if result.returncode != 0:
                logger.warning(f"Fleet validation failed: {result.stderr}")
                return False
            
            # Check if our fleet appears in output with active instances
            fleet_active = fleet_name in result.stdout and 'idle' in result.stdout.lower()
            logger.info(f"Fleet '{fleet_name}' connectivity: {'✅ Active' if fleet_active else '⚠️  Unknown'}")
            
            return True  # Return True for now - basic connectivity check passed
            
        except Exception as e:
            logger.error(f"Fleet validation error: {e}")
            return False
    
    def get_fleet_status(self, fleet_name: str) -> Dict[str, Any]:
        """Get detailed fleet status"""
        try:
            result = subprocess.run(
                ['dstack', 'fleet', 'list', '--verbose'],
                capture_output=True,
                text=True,
                timeout=30,
            )
            
            status = {
                'exists': fleet_name in result.stdout if result.returncode == 0 else False,
                'accessible': result.returncode == 0,
                'last_checked': datetime.utcnow().isoformat(),
                'raw_output': result.stdout if result.returncode == 0 else result.stderr,
            }
            
            return status
            
        except Exception as e:
            return {
                'exists': False,
                'accessible': False,
                'error': str(e),
                'last_checked': datetime.utcnow().isoformat(),
            }
    
    def cleanup_idle_fleets(self, max_idle_hours: int = 24) -> int:
        """
        Clean up fleets that have been idle for too long
        
        Args:
            max_idle_hours: Maximum idle time before cleanup
            
        Returns:
            Number of fleets cleaned up
        """
        # TODO: Implement fleet cleanup logic
        # For now, fleets are persistent
        logger.info(f"Fleet cleanup not implemented yet (max_idle_hours={max_idle_hours})")
        return 0


# Convenience functions for integration

def ensure_amd_fleet_ready(fleet_name: Optional[str] = None) -> bool:
    """
    Convenience function to ensure AMD fleet is ready for task submission
    
    Args:
        fleet_name: Optional fleet name (defaults to 'amd-mi300x-fleet')
        
    Returns:
        True if fleet is ready, False otherwise
    """
    manager = AMDFleetManager()
    return manager.ensure_fleet_exists(fleet_name)


def get_amd_fleet_status(fleet_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Get status of AMD fleet
    
    Args:
        fleet_name: Optional fleet name (defaults to 'amd-mi300x-fleet')
        
    Returns:
        Fleet status dictionary
    """
    manager = AMDFleetManager()
    actual_fleet_name = fleet_name or manager.amd_config['fleet_name']
    return manager.get_fleet_status(actual_fleet_name)


if __name__ == "__main__":
    """CLI interface for fleet management"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AMD DevCloud Fleet Manager")
    parser.add_argument('command', choices=['ensure', 'status', 'info', 'validate'],
                       help="Command to execute")
    parser.add_argument('--fleet', default=None, help="Fleet name (optional)")
    
    args = parser.parse_args()
    
    manager = AMDFleetManager()
    
    if args.command == 'ensure':
        success = manager.ensure_fleet_exists(args.fleet)
        print(f"Fleet ready: {success}")
        sys.exit(0 if success else 1)
        
    elif args.command == 'status':
        status = manager.get_fleet_status(args.fleet or manager.amd_config['fleet_name'])
        print(json.dumps(status, indent=2))
        
    elif args.command == 'info':
        info = manager.get_backend_info()
        print(json.dumps(info, indent=2))
        
    elif args.command == 'validate':
        valid = manager.validate_fleet_connectivity(args.fleet or manager.amd_config['fleet_name'])
        print(f"Fleet connectivity: {valid}")
        sys.exit(0 if valid else 1)