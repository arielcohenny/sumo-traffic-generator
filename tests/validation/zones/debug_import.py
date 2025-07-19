#!/usr/bin/env python3
"""
Debug script to test intelligent zones import
"""

import sys
import traceback

def test_imports():
    """Test all imports step by step"""
    print("Testing imports step by step...")
    
    try:
        print("1. Testing basic imports...")
        import xml.etree.ElementTree as ET
        from typing import Dict, List, Tuple, Optional, Set
        from pathlib import Path
        import logging
        import numpy as np
        import json
        from dataclasses import dataclass
        print("✅ Basic imports successful")
    except Exception as e:
        print(f"❌ Basic imports failed: {e}")
        return False
    
    try:
        print("2. Testing shapely...")
        from shapely.geometry import Polygon, Point, MultiPolygon
        from shapely.ops import unary_union
        print("✅ Shapely imports successful")
    except Exception as e:
        print(f"❌ Shapely imports failed: {e}")
        return False
    
    try:
        print("3. Testing optional imports...")
        try:
            import geopandas as gpd
            print("✅ geopandas available")
        except ImportError:
            print("⚠️  geopandas not available")
        
        try:
            import networkx as nx
            print("✅ networkx available")
        except ImportError:
            print("⚠️  networkx not available")
        
        try:
            import pandas as pd
            print("✅ pandas available")
        except ImportError:
            print("⚠️  pandas not available")
    except Exception as e:
        print(f"❌ Optional imports test failed: {e}")
        return False
    
    try:
        print("4. Testing intelligent zones import...")
        from src.network.intelligent_zones import IntelligentZoneGenerator
        print("✅ IntelligentZoneGenerator import successful")
    except Exception as e:
        print(f"❌ IntelligentZoneGenerator import failed: {e}")
        traceback.print_exc()
        return False
    
    try:
        print("5. Testing intelligent zones instantiation...")
        generator = IntelligentZoneGenerator(land_use_block_size_m=200.0)
        print("✅ IntelligentZoneGenerator instantiation successful")
    except Exception as e:
        print(f"❌ IntelligentZoneGenerator instantiation failed: {e}")
        traceback.print_exc()
        return False
    
    print("🎉 All tests passed!")
    return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)