#!/usr/bin/env python3
"""
Test CLI with intelligent zone generation
"""

import subprocess
import sys
from pathlib import Path

def test_cli_with_intelligent_zones():
    """Test the CLI with the new intelligent zone parameters"""
    print("Testing CLI with Intelligent Zone Generation")
    print("=" * 50)
    
    # Test command with new parameter
    cmd = [
        sys.executable, "-m", "src.cli",
        "--grid_dimension", "3",
        "--block_size_m", "150", 
        "--num_vehicles", "50",
        "--end-time", "300",
        "--land_use_block_size_m", "100",
        "--traffic_control", "fixed"  # Use fixed to avoid Tree Method setup complexity
    ]
    
    print(f"🚀 Running command: {' '.join(cmd)}")
    print()
    
    try:
        # Run the command
        result = subprocess.run(
            cmd,
            cwd="/Users/arielcohen/development/ariel_dev/sumo/Projects/sumo-traffic-generator",
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )
        
        print("📝 STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("\n⚠️  STDERR:")
            print(result.stderr)
        
        print(f"\n📊 Return code: {result.returncode}")
        
        # Check if zones file was created
        zones_file = Path("data/zones.poly.xml")
        if zones_file.exists():
            file_size = zones_file.stat().st_size
            print(f"✅ Zones file created: {zones_file} ({file_size} bytes)")
            
            # Read first few lines to check format
            with open(zones_file, 'r') as f:
                lines = f.readlines()[:10]
            print(f"📄 First few lines of zones file:")
            for i, line in enumerate(lines):
                print(f"   {i+1:>2}: {line.rstrip()}")
        else:
            print(f"❌ Zones file not found: {zones_file}")
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("❌ Command timed out after 2 minutes")
        return False
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False

if __name__ == "__main__":
    success = test_cli_with_intelligent_zones()
    exit(0 if success else 1)