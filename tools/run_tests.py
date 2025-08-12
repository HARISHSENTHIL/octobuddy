#!/usr/bin/env python3
"""
OctoBuddy Test Runner
Easy access to all wake word optimization and testing tools
"""

import os
import sys
import subprocess

def show_menu():
    """Show the test menu"""
    print("🧪 OCTOBUDDY WAKE WORD TEST SUITE")
    print("=" * 50)
    print("1. 🎯 Find Optimal Threshold (RECOMMENDED)")
    print("2. 🔬 Analyze Wake Word Model Structure")
    print("3. ⚡ Test Efficient Wake Detector")
    print("4. 🎤 Test Microphone Selection")
    print("5. 🚀 Run Push-to-Talk Mode")
    print("6. 📊 Quick Wake Word Test")
    print("7. ❌ Exit")
    print()

def run_test(choice):
    """Run the selected test"""
    test_dir = os.path.dirname(__file__)
    
    if choice == "1":
        print("🎯 Running Threshold Optimizer...")
        subprocess.run([sys.executable, "find_optimal_threshold.py"], cwd=test_dir)
    
    elif choice == "2":
        print("🔬 Running Model Analysis...")
        subprocess.run([sys.executable, "analyze_wake_model.py"], cwd=test_dir)
    
    elif choice == "3":
        print("⚡ Testing Efficient Wake Detector...")
        subprocess.run([sys.executable, "efficient_wake_detector.py"], cwd=test_dir)
    
    elif choice == "4":
        print("🎤 Running Microphone Selector...")
        subprocess.run([sys.executable, "mic_selector.py"], cwd=test_dir)
    
    elif choice == "5":
        print("🚀 Starting Push-to-Talk Mode...")
        subprocess.run([sys.executable, "run_push_to_talk.py"], cwd=test_dir)
    
    elif choice == "6":
        print("📊 Quick Wake Word Test...")
        quick_test()
    
    elif choice == "7":
        print("👋 Goodbye!")
        return False
    
    else:
        print("❌ Invalid choice. Please try again.")
    
    return True

def quick_test():
    """Quick wake word test"""
    try:
        sys.path.append(os.path.dirname(test_dir))
        from wake_improved import DirectWakeONNX, wait_for_wake
        from main import INPUT_DEVICE_INDEX
        
        print("📊 QUICK WAKE WORD TEST")
        print("=" * 30)
        
        model_path = "/Users/harish/Documents/WORK/octobuddy/Hey_octobuddy.onnx"
        model = DirectWakeONNX(model_path)
        
        print("Say 'Hey Octobuddy' (10 second test)...")
        
        detected, score = wait_for_wake(
            model,
            threshold=0.005,
            input_device_index=INPUT_DEVICE_INDEX,
            timeout_sec=10,
            verbose=True
        )
        
        if detected:
            print(f"✅ SUCCESS! Score: {score:.3f}")
        
    except TimeoutError:
        print("⏰ No wake word detected in 10 seconds")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_dir = os.path.dirname(__file__)
    
    print("🔧 OctoBuddy Wake Word Testing & Optimization")
    print("This tool helps you get the best wake word performance.\n")
    
    while True:
        try:
            show_menu()
            choice = input("Enter your choice (1-7): ").strip()
            print()
            
            if not run_test(choice):
                break
                
            print("\n" + "="*50)
            input("Press ENTER to continue...")
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 Tests cancelled.")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            input("Press ENTER to continue...")
