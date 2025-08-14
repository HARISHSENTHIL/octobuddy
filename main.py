#!/usr/bin/env python3
"""
OctoBuddy - Ubuntu Voice Assistant
Main launcher script with dual wake word detection
"""

import sys
import os
from pathlib import Path

def main():
    """Main launcher function"""
    # Add the core module to path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root / "octobuddy_core"))
    
    # Change to project root for relative paths to work
    os.chdir(project_root)
    
    # Import and run assistant
    try:
        from assistant import main as run_assistant
        print("🚀 Starting OctoBuddy Voice Assistant...")
        print("🎯 Dual wake words: 'Hey Octobuddy' (start) | 'Bye Octobuddy' (exit)")
        run_assistant()
    except KeyboardInterrupt:
        print("\n👋 OctoBuddy stopped.")
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()
