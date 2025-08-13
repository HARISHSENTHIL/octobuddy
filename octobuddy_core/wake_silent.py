#!/usr/bin/env python3
"""
Ubuntu-Optimized Wake Word Detection
Uses openWakeWord's built-in Speex noise suppression for optimal performance
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, Dict
import time
import numpy as np
import sounddevice as sd
import openwakeword
from openwakeword.model import Model

def _rms(x: np.ndarray) -> float:
    """Calculate RMS energy of audio"""
    if x.size == 0:
        return 0.0
    x = x.astype(np.float64, copy=False)
    return float(np.sqrt(np.mean(np.square(x))))

def create_progress_bar(score: float, threshold: float, complete: bool = False) -> str:
    """Create visual progress bar for wake word detection"""
    if complete:
        return "█████████████████████ ✅"
    
    max_display = threshold * 3
    progress = min(score / max_display, 1.0)
    filled = int(progress * 20)
    empty = 20 - filled
    bar = "█" * filled + "░" * empty
    
    if score >= threshold * 0.8:
        indicator = " 🔥"
    elif score >= threshold * 0.5:
        indicator = " 🟡"
    else:
        indicator = " 🔵"
    
    return bar + indicator

class UbuntuWakeDetector:
    """
    Ubuntu-optimized wake word detector using openWakeWord
    Leverages built-in Speex noise suppression for better performance
    """
    def __init__(self, model_path: str, enable_speex: bool = True):
        self.model_path = str(Path(model_path))
        self.enable_speex = enable_speex
        
        # Initialize openWakeWord model with Speex noise suppression
        try:
            # Try new API first (openWakeWord 0.4.0+)
            self.model = Model(
                wakeword_models=[self.model_path],
                enable_speex_noise_suppression=enable_speex,
                vad_threshold=0.5  # Voice Activity Detection threshold
            )
            print(f"[wakeword] ✅ openWakeWord model loaded with Speex: {enable_speex}")
        except TypeError as e:
            # Fallback to older API
            try:
                self.model = Model(
                    [self.model_path],
                    enable_speex_noise_suppression=enable_speex
                )
                print(f"[wakeword] ✅ openWakeWord model loaded (legacy API) with Speex: {enable_speex}")
            except Exception as e2:
                print(f"[wakeword] ❌ Failed to load openWakeWord model with both APIs: {e2}")
                raise
        except Exception as e:
            print(f"[wakeword] ❌ Failed to load openWakeWord model: {e}")
            raise

    def predict(self, audio_1s: np.ndarray) -> Dict[str, float]:
        """Predict wake word score using openWakeWord"""
        try:
            # Ensure audio is 16-bit 16kHz PCM
            if audio_1s.dtype != np.int16:
                audio_1s = (audio_1s * 32767).astype(np.int16)
            
            # Get prediction from openWakeWord
            result = self.model.predict(audio_1s)
            
            # Extract score for our model
            model_name = Path(self.model_path).stem
            score = result.get(model_name, 0.0)
            
            return {model_name: float(score)}
        except Exception as e:
            print(f"[wakeword] Prediction error: {e}")
            return {"hey_octobuddy": 0.0}

def wait_for_wake_ubuntu(
    model: UbuntuWakeDetector,
    threshold: float = 0.005,
    samplerate: int = 16000,
    input_device_index: Optional[int] = None,
    timeout_sec: Optional[float] = None,
    print_startup: bool = True
) -> Tuple[str, float]:
    """
    Ubuntu-optimized wake word detection using openWakeWord
    Leverages built-in Speex noise suppression and VAD
    """
    
    # Audio streaming setup
    chunk_ms = 1000  # 1-second chunks for openWakeWord compatibility
    blocksize = int(samplerate * (chunk_ms / 1000.0))
    blocksize = max(blocksize, 1024)
    
    # Audio stream configuration
    kwargs = {
        "channels": 1,
        "samplerate": samplerate,
        "blocksize": blocksize,
        "dtype": "int16",  # openWakeWord expects int16
    }
    if input_device_index is not None:
        kwargs["device"] = input_device_index

    if print_startup:
        print("[wakeword] 🔧 Initializing Ubuntu-optimized wake word detection...")
        print(f"[wakeword] 🎯 Using openWakeWord with Speex noise suppression")
        print(f"[wakeword] 👂 Listening for wake word (threshold: {threshold})")
        print(f"[wakeword] 🔇 Running silently until detection...")

    start_t = time.time()
    last_fire = 0.0
    consecutive_detections = 0
    required_consecutive = 1

    with sd.InputStream(**kwargs) as stream:
        while True:
            if timeout_sec is not None and (time.time() - start_t) > timeout_sec:
                raise TimeoutError("wake-word wait timed out")

            # Read audio chunk
            audio, _ = stream.read(blocksize)
            audio = np.squeeze(audio).astype(np.int16, copy=False)

            # Skip if audio is too quiet (efficiency)
            if _rms(audio.astype(np.float32)) < 0.001:
                time.sleep(0.1)
                continue

            # Predict wake word score
            preds = model.predict(audio)
            model_name = list(preds.keys())[0]
            score = preds[model_name]

            # Detection logic with visual progress
            current_time = time.time()
            if score >= threshold and (current_time - last_fire) >= 1.0:
                consecutive_detections += 1
                
                if consecutive_detections == 1:
                    progress_bar = create_progress_bar(score, threshold)
                    print(f"[wakeword] 🔥 Detecting... {progress_bar} {score:.3f}")
                
                if consecutive_detections >= required_consecutive:
                    final_bar = create_progress_bar(score, threshold, complete=True)
                    print(f"[wakeword] ✅ CONFIRMED! {final_bar} {score:.3f}")
                    last_fire = current_time
                    return model_name, score
                    
                time.sleep(0.05)
            else:
                consecutive_detections = 0
                
                # Show progress when speaking but not detecting
                if score > 0.001:
                    progress_bar = create_progress_bar(score, threshold)
                    print(f"[wakeword] 👂 Listening... {progress_bar} {score:.3f}", end='\r')

            time.sleep(0.05)

# Aliases for compatibility
DirectWakeONNX = UbuntuWakeDetector
wait_for_wake = wait_for_wake_ubuntu
