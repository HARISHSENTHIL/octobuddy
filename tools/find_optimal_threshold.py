#!/usr/bin/env python3
"""
Find Optimal Wake Word Threshold
Systematic approach to find the best threshold for your voice and microphone
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import soundfile as sf
from wake_improved import DirectWakeONNX
from main import record_until_silence, INPUT_DEVICE_INDEX
import time

def record_wake_word_samples(num_samples=5):
    """Record multiple 'Hey Octobuddy' samples and analyze scores"""
    print("🎤 RECORDING WAKE WORD SAMPLES FOR ANALYSIS")
    print("=" * 60)
    print(f"Recording {num_samples} samples of 'Hey Octobuddy'")
    print("This will help us find the perfect threshold for your voice.\n")
    
    model_path = "/Users/harish/Documents/WORK/octobuddy/Hey_octobuddy.onnx"
    model = DirectWakeONNX(model_path)
    
    all_scores = []
    
    for i in range(num_samples):
        print(f"📼 Sample {i+1}/{num_samples}")
        print("   Say 'Hey Octobuddy' clearly when recording starts...")
        
        input("   Press ENTER to start recording...")
        
        wav_file = f"test/wake_sample_{i+1}.wav"
        
        # Record with optimized settings
        ok = record_until_silence(
            wav_file,
            device_index=INPUT_DEVICE_INDEX,
            sr=16000,
            block=512,
            baseline_sec=0.5,
            start_gate_mul=1.1,  # Lower for easier triggering
            start_min_ms=80,     # Shorter minimum
            silence_gate_mul=1.2,
            end_sil_ms=500,      # Shorter silence
            max_record_sec=4.0
        )
        
        if not ok:
            print("   ❌ Recording failed, skipping...")
            continue
        
        # Analyze the recording
        try:
            audio, sr = sf.read(wav_file)
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
            
            print(f"   ✅ Recorded {len(audio)/16000:.2f}s of audio")
            
            # Test sliding windows to find best score
            sample_scores = []
            
            if len(audio) >= 16000:
                # Test overlapping 1-second windows
                for start in range(0, len(audio) - 16000 + 1, 4000):  # 0.25s steps
                    segment = audio[start:start+16000].astype(np.float32)
                    result = model.predict(segment)
                    score = result['hey_octobuddy']
                    sample_scores.append(score)
                
                best_score = max(sample_scores)
                avg_score = np.mean(sample_scores)
                
                print(f"   📊 Best score: {best_score:.4f}")
                print(f"   📊 Average score: {avg_score:.4f}")
                
                all_scores.extend(sample_scores)
            else:
                print("   ⚠️ Recording too short for analysis")
        
        except Exception as e:
            print(f"   ❌ Analysis failed: {e}")
        
        print()
    
    return all_scores

def analyze_scores_and_recommend_threshold(scores):
    """Analyze all scores and recommend optimal threshold"""
    if not scores:
        print("❌ No scores to analyze!")
        return None
    
    scores = np.array(scores)
    
    print("📊 SCORE ANALYSIS")
    print("=" * 40)
    print(f"Total samples analyzed: {len(scores)}")
    print(f"Maximum score: {np.max(scores):.6f}")
    print(f"Average score: {np.mean(scores):.6f}")
    print(f"Median score: {np.median(scores):.6f}")
    print(f"Standard deviation: {np.std(scores):.6f}")
    print(f"95th percentile: {np.percentile(scores, 95):.6f}")
    print(f"75th percentile: {np.percentile(scores, 75):.6f}")
    
    # Find score distribution
    high_scores = scores[scores > 0.01]
    medium_scores = scores[(scores > 0.001) & (scores <= 0.01)]
    low_scores = scores[scores <= 0.001]
    
    print(f"\nScore Distribution:")
    print(f"  High scores (>0.01): {len(high_scores)} ({len(high_scores)/len(scores)*100:.1f}%)")
    print(f"  Medium scores (0.001-0.01): {len(medium_scores)} ({len(medium_scores)/len(scores)*100:.1f}%)")
    print(f"  Low scores (≤0.001): {len(low_scores)} ({len(low_scores)/len(scores)*100:.1f}%)")
    
    # Recommend threshold
    max_score = np.max(scores)
    p95_score = np.percentile(scores, 95)
    p75_score = np.percentile(scores, 75)
    
    print(f"\n💡 THRESHOLD RECOMMENDATIONS:")
    
    if max_score > 0.1:
        recommended = max_score * 0.6
        confidence = "HIGH"
        print(f"✅ Excellent detection possible!")
        print(f"   Recommended threshold: {recommended:.4f}")
        print(f"   Confidence: {confidence}")
    elif max_score > 0.01:
        recommended = max_score * 0.7
        confidence = "MEDIUM"
        print(f"🟡 Good detection possible with tuning")
        print(f"   Recommended threshold: {recommended:.4f}")
        print(f"   Confidence: {confidence}")
        print(f"   Alternative (conservative): {p75_score:.4f}")
    elif max_score > 0.001:
        recommended = max_score * 0.8
        confidence = "LOW"
        print(f"🟠 Weak detection - model may need retraining")
        print(f"   Try threshold: {recommended:.4f}")
        print(f"   Confidence: {confidence}")
        print(f"   Consider: Speaking louder, getting closer to mic")
    else:
        recommended = 0.001
        confidence = "VERY LOW"
        print(f"❌ Poor detection - significant issues")
        print(f"   Model likely incompatible with your voice/setup")
        print(f"   Try threshold: {recommended:.4f} (may cause false positives)")
        print(f"   Recommend: Model retraining or different wake word")
    
    return recommended

def test_recommended_threshold(threshold):
    """Test the recommended threshold in real-time"""
    print(f"\n🧪 TESTING RECOMMENDED THRESHOLD: {threshold:.4f}")
    print("=" * 50)
    
    try:
        # Import the improved wake detector
        from wake_improved import wait_for_wake
        
        model_path = "/Users/harish/Documents/WORK/octobuddy/Hey_octobuddy.onnx"
        model = DirectWakeONNX(model_path)
        
        print(f"Say 'Hey Octobuddy' to test the threshold...")
        print(f"Testing for 20 seconds...\n")
        
        detected, score = wait_for_wake(
            model,
            threshold=threshold,
            input_device_index=INPUT_DEVICE_INDEX,
            timeout_sec=20,
            verbose=True
        )
        
        if detected:
            print(f"\n✅ SUCCESS! Wake word detected with score: {score:.4f}")
            print(f"💡 This threshold ({threshold:.4f}) works for your voice!")
            return True
        
    except TimeoutError:
        print(f"\n⏰ TIMEOUT: No detection in 20 seconds")
        print(f"💡 Try a lower threshold or speak louder/clearer")
        return False
    except KeyboardInterrupt:
        print(f"\n🛑 Test cancelled by user")
        return False
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        return False

def update_main_py_threshold(threshold):
    """Update the threshold in main.py"""
    import re
    from pathlib import Path
    
    try:
        main_file = Path("main.py")
        if not main_file.exists():
            print("❌ main.py not found!")
            return False
        
        content = main_file.read_text()
        
        # Find and replace threshold
        pattern = r'threshold=[\d\.]+'
        replacement = f'threshold={threshold:.4f}'
        
        if re.search(pattern, content):
            new_content = re.sub(pattern, replacement, content)
            main_file.write_text(new_content)
            print(f"✅ Updated main.py with threshold: {threshold:.4f}")
            return True
        else:
            print(f"❌ Could not find threshold setting in main.py")
            print(f"💡 Manually update the threshold to: {threshold:.4f}")
            return False
            
    except Exception as e:
        print(f"❌ Error updating main.py: {e}")
        return False

def main():
    """Main threshold optimization workflow"""
    print("🎯 OCTOBUDDY THRESHOLD OPTIMIZER")
    print("=" * 60)
    print("This tool will help you find the perfect wake word threshold")
    print("for your voice and microphone setup.\n")
    
    # Step 1: Record samples
    print("STEP 1: Recording wake word samples")
    print("-" * 40)
    scores = record_wake_word_samples(num_samples=5)
    
    if not scores:
        print("❌ No valid recordings. Please try again.")
        return
    
    # Step 2: Analyze and recommend
    print("\nSTEP 2: Analyzing scores and recommending threshold")
    print("-" * 40)
    recommended_threshold = analyze_scores_and_recommend_threshold(scores)
    
    if recommended_threshold is None:
        return
    
    # Step 3: Test recommendation
    print(f"\nSTEP 3: Testing recommended threshold")
    print("-" * 40)
    
    test_it = input(f"Test threshold {recommended_threshold:.4f} in real-time? (y/n): ").lower().strip()
    
    if test_it in ['y', 'yes', '']:
        success = test_recommended_threshold(recommended_threshold)
        
        if success:
            # Step 4: Update configuration
            update_config = input(f"\nUpdate main.py with this threshold? (y/n): ").lower().strip()
            if update_config in ['y', 'yes', '']:
                update_main_py_threshold(recommended_threshold)
        else:
            # Suggest lower threshold
            lower_threshold = recommended_threshold * 0.5
            print(f"\n💡 Consider trying a lower threshold: {lower_threshold:.4f}")
            
            try_lower = input(f"Test lower threshold {lower_threshold:.4f}? (y/n): ").lower().strip()
            if try_lower in ['y', 'yes', '']:
                success = test_recommended_threshold(lower_threshold)
                if success:
                    update_config = input(f"\nUpdate main.py with threshold {lower_threshold:.4f}? (y/n): ").lower().strip()
                    if update_config in ['y', 'yes', '']:
                        update_main_py_threshold(lower_threshold)
    
    print(f"\n🎉 THRESHOLD OPTIMIZATION COMPLETE!")
    print(f"You can now run: python3 main.py")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n👋 Threshold optimization cancelled.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
