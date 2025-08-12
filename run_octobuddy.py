#!/usr/bin/env python3
"""
OctoBuddy Launcher
Simple script to run OctoBuddy from the reorganized structure
"""

import sys
import os
from pathlib import Path

# Add the core module to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "octobuddy_core"))

# Change to project root for relative paths to work
os.chdir(project_root)

# Import and run main
try:
    from main import main
    print("🚀 Starting OctoBuddy Educational Assistant...")
    main()
except KeyboardInterrupt:
    print("\n👋 OctoBuddy stopped.")
except Exception as e:
    print(f"❌ Error: {e}")
