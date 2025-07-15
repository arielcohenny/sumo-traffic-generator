#!/usr/bin/env python3
"""
Quick test for the intelligent zones system
"""

import sys
import os
import subprocess
from pathlib import Path

def run_quick_test():
    """Run a quick test of the CLI with intelligent zones"""
    print("🧪 Testing CLI with Intelligent Zones System")
    print("=" * 50)
    
    # Ensure we're in the right directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    print(f"📁 Working directory: {os.getcwd()}")
    
    # Test command - very small scenario for quick testing
    cmd = [
        sys.executable, "-m", "src.cli",
        "--grid_dimension", "3",  # Small 3x3 grid
        "--block_size_m", "100",  # Small blocks
        "--num_vehicles", "5",    # Very few vehicles
        "--end-time", "60",       # 1 minute simulation
        "--land_use_block_size_m", "50",  # Small zone resolution
        "--traffic_control", "fixed"  # Use fixed to avoid Tree Method complexity
    ]
    
    print(f"🚀 Running command:")
    print(f"    {' '.join(cmd)}")
    print()
    
    try:
        # Run the command with timeout
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60  # 1 minute timeout
        )
        
        print("📊 Return code:", result.returncode)
        print()
        
        print("📝 STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("\n⚠️  STDERR:")
            print(result.stderr)
        
        # Check for success indicators
        if result.returncode == 0:
            print("\n✅ Command completed successfully!")
        else:
            print(f"\n❌ Command failed with return code {result.returncode}")
        
        # Check if zones file was created
        zones_file = Path("data/zones.poly.xml")
        if zones_file.exists():
            file_size = zones_file.stat().st_size
            print(f"✅ Zones file created: {zones_file} ({file_size} bytes)")
        else:
            print(f"❌ Zones file not found: {zones_file}")
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("❌ Command timed out after 1 minute")
        return False
    except Exception as e:
        print(f"❌ Error running command: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_quick_test()
    sys.exit(0 if success else 1)