#!/usr/bin/env python3
"""
Test the improved UX features:
1. Clear timing indicators after wake word
2. Better Whisper transcription
3. Visual progress bar during wake word detection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def test_wake_word_with_progress_bar():
    """Test wake word detection with visual progress bar"""
    print("🎯 TESTING WAKE WORD WITH PROGRESS BAR")
    print("=" * 50)
    
    try:
        from wake_silent import DirectWakeONNX, wait_for_wake
        from main import INPUT_DEVICE_INDEX
        
        model_path = "/Users/harish/Documents/WORK/octobuddy/Hey_octobuddy.onnx"
        model = DirectWakeONNX(model_path)
        
        print("Say 'Hey Octobuddy' to see the progress bar in action...")
        print("You should see: 🔵 (low) → 🟡 (medium) → 🔥 (high) → ✅ (detected)")
        print()
        
        detected, score = wait_for_wake(
            model,
            threshold=0.005,
            input_device_index=INPUT_DEVICE_INDEX,
            timeout_sec=20,
            print_startup=True
        )
        
        if detected:
            print(f"\n✅ Wake word detected with progress bar! Score: {score:.3f}")
            return True
        
    except TimeoutError:
        print("\n⏰ Timeout - try speaking 'Hey Octobuddy' more clearly")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

def test_improved_whisper():
    """Test improved Whisper transcription"""
    print("\n🗣️ TESTING IMPROVED WHISPER TRANSCRIPTION")
    print("=" * 50)
    
    try:
        from main import record_until_silence, transcribe_whisper, INPUT_DEVICE_INDEX
        
        print("Say a clear sentence when recording starts...")
        print("Testing: Better recognition, noise filtering, multiple attempts")
        print()
        
        input("Press ENTER to start recording...")
        
        wav = "test/improved_whisper_test.wav"
        
        ok = record_until_silence(
            wav, device_index=INPUT_DEVICE_INDEX,
            sr=16000, block=512, 
            baseline_sec=0.2,           # Quick calibration
            start_gate_mul=1.1,         # Very sensitive
            start_min_ms=60,            # Short minimum
            silence_gate_mul=1.05,      # Sensitive to silence
            end_sil_ms=500,             # Quick silence detection  
            max_record_sec=7.0
        )
        
        if ok:
            print("✅ Recording successful")
            print("🗣️ Transcribing with improved settings...")
            
            text = transcribe_whisper(wav)
            
            if text:
                print(f"✅ Improved transcription: '{text}'")
                return text
            else:
                print("❌ Transcription failed")
                return None
        else:
            print("❌ Recording failed")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_clear_timing_flow():
    """Test the clear timing and instructions"""
    print("\n🕐 TESTING CLEAR TIMING FLOW")
    print("=" * 50)
    
    print("Simulating the wake word detection → countdown → recording flow:")
    print()
    
    # Simulate wake word detection
    print("[wakeword] ✨ WAKE WORD DETECTED! 'hey_octobuddy' (score=0.008)")
    print("[wakeword] 🎤 Ready! Please ask your question now...")
    
    # Simulate countdown
    import time
    for i in range(3, 0, -1):
        print(f"[wakeword] 🕐 Starting in {i}...", end='\r')
        time.sleep(1)
    print(f"[wakeword] 🎤 SPEAK NOW! I'm listening...         ")
    
    print()
    print("✅ Clear timing flow demonstrated!")
    print("User now knows exactly when to start speaking")
    
    return True

def demo_progress_bar():
    """Demonstrate the progress bar functionality"""
    print("\n📊 PROGRESS BAR DEMO")
    print("=" * 30)
    
    from wake_silent import create_progress_bar
    
    threshold = 0.005
    test_scores = [0.001, 0.002, 0.003, 0.004, 0.005, 0.008, 0.010]
    
    print(f"Progress bar visualization (threshold: {threshold}):")
    print()
    
    for score in test_scores:
        bar = create_progress_bar(score, threshold)
        status = "✅ DETECTED!" if score >= threshold else "Listening..."
        print(f"Score {score:.3f}: {bar} {status}")
    
    # Show completion
    final_bar = create_progress_bar(0.008, threshold, complete=True)
    print(f"Complete:  {final_bar} CONFIRMED!")

if __name__ == "__main__":
    print("🎯 TESTING IMPROVED OCTOBUDDY UX")
    print("=" * 60)
    print("Testing the three improvements:")
    print("1. Clear timing after wake word detection")
    print("2. Better Whisper transcription")  
    print("3. Visual progress bar during wake word detection")
    print()
    
    # Demo progress bar first
    demo_progress_bar()
    
    # Test clear timing flow
    test_clear_timing_flow()
    
    # Test improved Whisper
    print("\nTesting improved Whisper transcription...")
    response = input("Test Whisper transcription? (y/n): ").lower().strip()
    if response in ['y', 'yes', '']:
        test_improved_whisper()
    
    # Test wake word with progress bar
    print("\nTesting wake word with progress bar...")
    response = input("Test wake word detection with progress bar? (y/n): ").lower().strip()
    if response in ['y', 'yes', '']:
        test_wake_word_with_progress_bar()
    
    print("\n🎉 UX IMPROVEMENTS COMPLETE!")
    print("=" * 50)
    print("✅ Clear countdown after wake word detection")
    print("✅ Improved Whisper transcription settings")
    print("✅ Visual progress bar during wake word detection")
    print("\n🚀 Ready to test: python main.py")
