#!/usr/bin/env python3
"""
Microphone Selector for OctoBuddy
Helps you choose the best microphone for wake word detection
"""

import sounddevice as sd
import numpy as np
import time
from pathlib import Path

def list_all_audio_devices():
    """List all audio devices with detailed information"""
    print("🎤 COMPLETE AUDIO DEVICE LIST")
    print("=" * 60)
    
    try:
        devs = sd.query_devices()
        input_devices = []
        
        for i, d in enumerate(devs):
            device_type = []
            if d.get('max_input_channels', 0) > 0:
                device_type.append(f"📥 INPUT ({d['max_input_channels']} ch)")
                input_devices.append(i)
            if d.get('max_output_channels', 0) > 0:
                device_type.append(f"📤 OUTPUT ({d['max_output_channels']} ch)")
            
            if device_type:
                print(f"#{i:2d}: {d['name']}")
                print(f"     Type: {' | '.join(device_type)}")
                print(f"     Sample Rate: {int(d.get('default_samplerate', 0))} Hz")
                print(f"     Host API: {d.get('hostapi', 'unknown')}")
                print()
        
        print(f"🔧 System Default Input: #{sd.default.device[0]}")
        print(f"🔧 System Default Output: #{sd.default.device[1]}")
        
        return input_devices
        
    except Exception as e:
        print(f"❌ Error listing devices: {e}")
        return []

def test_microphone_quality(device_idx, duration=3.0):
    """Test microphone quality by recording a sample"""
    print(f"\n🎤 Testing Device #{device_idx}...")
    print(f"   Speak clearly for {duration} seconds...")
    
    try:
        # Record sample
        sample_rate = 16000
        audio = sd.rec(
            int(duration * sample_rate), 
            samplerate=sample_rate, 
            channels=1, 
            device=device_idx,
            dtype='float32'
        )
        
        print("   Recording... speak now!")
        sd.wait()
        
        # Analyze quality
        audio = audio.flatten()
        rms = np.sqrt(np.mean(audio**2))
        peak = np.max(np.abs(audio))
        snr_estimate = 20 * np.log10(rms / (np.std(audio) + 1e-10))
        
        print(f"   📊 Audio Quality Analysis:")
        print(f"      RMS Level: {rms:.4f}")
        print(f"      Peak Level: {peak:.4f}")
        print(f"      SNR Estimate: {snr_estimate:.1f} dB")
        
        # Quality rating
        if rms > 0.01 and peak < 0.95:
            quality = "✅ EXCELLENT"
        elif rms > 0.005:
            quality = "🟡 GOOD"
        elif rms > 0.001:
            quality = "🟠 FAIR"
        else:
            quality = "❌ POOR"
        
        print(f"      Quality: {quality}")
        
        return rms, quality
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return 0.0, "❌ ERROR"

def interactive_mic_selection():
    """Interactive microphone selection process"""
    print("🎯 OCTOBUDDY MICROPHONE SELECTOR")
    print("=" * 60)
    
    # List all devices
    input_devices = list_all_audio_devices()
    
    if not input_devices:
        print("❌ No input devices found!")
        return None
    
    print(f"\n📝 Found {len(input_devices)} input device(s)")
    
    # Test each microphone
    print("\n🧪 MICROPHONE QUALITY TEST")
    print("=" * 40)
    
    test_results = {}
    
    for device_idx in input_devices:
        try:
            device_info = sd.query_devices(device_idx)
            print(f"\n🎤 Testing: {device_info['name']}")
            
            # Ask user if they want to test this device
            response = input(f"   Test this microphone? (y/n/skip): ").lower().strip()
            
            if response == 'skip':
                break
            elif response in ['y', 'yes', '']:
                rms, quality = test_microphone_quality(device_idx, duration=3.0)
                test_results[device_idx] = {
                    'name': device_info['name'],
                    'rms': rms,
                    'quality': quality
                }
            else:
                print("   ⏭️ Skipped")
                
        except Exception as e:
            print(f"   ❌ Error testing device: {e}")
    
    # Show results and recommendations
    if test_results:
        print("\n🏆 TEST RESULTS & RECOMMENDATIONS")
        print("=" * 50)
        
        best_device = None
        best_rms = 0.0
        
        for device_idx, result in test_results.items():
            print(f"#{device_idx}: {result['name']}")
            print(f"   Quality: {result['quality']}")
            print(f"   RMS Level: {result['rms']:.4f}")
            
            if result['rms'] > best_rms:
                best_rms = result['rms']
                best_device = device_idx
            print()
        
        if best_device is not None:
            print(f"🎯 RECOMMENDED: Device #{best_device}")
            print(f"   {test_results[best_device]['name']}")
            print(f"   Quality: {test_results[best_device]['quality']}")
            
            # Ask if user wants to update config
            response = input(f"\nUpdate OctoBuddy to use Device #{best_device}? (y/n): ").lower().strip()
            if response in ['y', 'yes', '']:
                update_config(best_device)
                return best_device
    
    return None

def update_config(device_idx):
    """Update main.py with the selected device index"""
    try:
        main_file = Path("main.py")
        if not main_file.exists():
            print("❌ main.py not found!")
            return False
        
        # Read current content
        content = main_file.read_text()
        
        # Update INPUT_DEVICE_INDEX
        import re
        pattern = r'INPUT_DEVICE_INDEX\s*=\s*\d+'
        replacement = f'INPUT_DEVICE_INDEX = {device_idx}'
        
        if re.search(pattern, content):
            new_content = re.sub(pattern, replacement, content)
            main_file.write_text(new_content)
            print(f"✅ Updated main.py: INPUT_DEVICE_INDEX = {device_idx}")
            return True
        else:
            print("❌ Could not find INPUT_DEVICE_INDEX in main.py")
            print(f"💡 Please manually set: INPUT_DEVICE_INDEX = {device_idx}")
            return False
            
    except Exception as e:
        print(f"❌ Error updating config: {e}")
        return False

def quick_wake_word_test(device_idx):
    """Quick test of wake word detection with selected microphone"""
    try:
        print(f"\n🔊 QUICK WAKE WORD TEST")
        print(f"Testing with Device #{device_idx}")
        
        from wake import DirectWakeONNX, wait_for_wake
        
        model_path = "/Users/harish/Documents/WORK/octobuddy/Hey_octobuddy.onnx"
        model = DirectWakeONNX(model_path)
        
        print("Say 'Hey Octobuddy' now...")
        
        name, score = wait_for_wake(
            model,
            threshold=0.01,  # Low threshold for testing
            input_device_index=device_idx,
            timeout_sec=10,
            verbose=True,
        )
        
        print(f"✅ Wake word detected: '{name}' (score: {score:.3f})")
        return True
        
    except Exception as e:
        print(f"❌ Wake word test failed: {e}")
        return False

if __name__ == "__main__":
    try:
        selected_device = interactive_mic_selection()
        
        if selected_device is not None:
            print(f"\n🎉 Setup complete! Device #{selected_device} selected.")
            
            # Optional wake word test
            response = input("\nTest wake word detection with selected microphone? (y/n): ").lower().strip()
            if response in ['y', 'yes', '']:
                quick_wake_word_test(selected_device)
                
            print("\n🚀 Ready to run OctoBuddy!")
            print("   Run: python3 main.py")
        else:
            print("\n⚠️ No microphone selected. Using default device.")
            
    except KeyboardInterrupt:
        print("\n\n👋 Microphone selection cancelled.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
