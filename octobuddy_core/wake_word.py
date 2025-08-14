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

class DualWakeDetector:
    """
    Dual wake word detector for entry and exit commands
    Monitors both "Hey Octobuddy" (entry) and "Bye Octobuddy" (exit) simultaneously
    """
    def __init__(self, entry_model_path: str, exit_model_path: str, enable_speex: bool = True):
        self.entry_model_path = str(Path(entry_model_path))
        self.exit_model_path = str(Path(exit_model_path))
        self.enable_speex = enable_speex
        
        try:
            # Load both models in a single openWakeWord instance
            model_paths = [self.entry_model_path, self.exit_model_path]
            
            # Try modern API first (keyword argument)
            self.model = Model(
                wakeword_models=model_paths,
                enable_speex_noise_suppression=enable_speex,
                vad_threshold=0.5
            )
            print(f"[wakeword] ✅ Dual wake word models loaded with Speex: {enable_speex}")
            print(f"[wakeword] 📥 Entry model: {Path(self.entry_model_path).name}")
            print(f"[wakeword] 📤 Exit model: {Path(self.exit_model_path).name}")
        except TypeError as e:
            # Fall back to legacy API (positional argument)
            try:
                self.model = Model(
                    model_paths,
                    enable_speex_noise_suppression=enable_speex
                )
                print(f"[wakeword] ✅ Dual wake word models loaded (legacy API) with Speex: {enable_speex}")
                print(f"[wakeword] 📥 Entry model: {Path(self.entry_model_path).name}")
                print(f"[wakeword] 📤 Exit model: {Path(self.exit_model_path).name}")
            except Exception as e2:
                print(f"[wakeword] ❌ Failed to load dual wake word models with both APIs: {e2}")
                raise
        except Exception as e:
            print(f"[wakeword] ❌ Failed to load dual wake word models: {e}")
            raise
    
    def predict(self, audio_data: np.ndarray) -> Dict[str, float]:
        """
        Predict wake word scores for both entry and exit models
        Returns dict with model names as keys and scores as values
        """
        try:
            # Ensure audio is in int16 format as expected by openWakeWord
            if audio_data.dtype != np.int16:
                audio_data = (audio_data * 32767).astype(np.int16)
            
            # Get prediction from openWakeWord
            result = self.model.predict(audio_data)
            
            # Extract scores for both models
            entry_name = Path(self.entry_model_path).stem
            exit_name = Path(self.exit_model_path).stem
            
            entry_score = result.get(entry_name, 0.0)
            exit_score = result.get(exit_name, 0.0)
            
            return {
                'entry': float(entry_score),
                'exit': float(exit_score),
                entry_name: float(entry_score),
                exit_name: float(exit_score)
            }
        except Exception as e:
            print(f"[wakeword] Dual prediction error: {e}")
            return {'entry': 0.0, 'exit': 0.0}

def wait_for_dual_wake(
    dual_model: DualWakeDetector,
    entry_threshold: float = 0.020,
    exit_threshold: float = 0.025,
    samplerate: int = 16000,
    input_device_index: Optional[int] = None,
    timeout_sec: Optional[float] = None,
    print_startup: bool = True,
    conversation_mode: bool = False  # When True, check for exit wake word only
) -> Tuple[str, float, str]:  # Returns (wake_type, score, model_name)
    """
    Dual wake word detection for entry and exit commands
    Returns wake_type: 'entry' for Hey Octobuddy, 'exit' for Bye Octobuddy
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
        if conversation_mode:
            print("[wakeword] 🔧 Monitoring for exit wake word during conversation...")
            print(f"[wakeword] 📤 Say 'Bye Octobuddy' to end conversation (threshold: {exit_threshold})")
        else:
            print("[wakeword] 🔧 Initializing dual wake word detection...")
            print(f"[wakeword] 📥 Entry: 'Hey Octobuddy' (threshold: {entry_threshold})")
            print(f"[wakeword] 📤 Exit: 'Bye Octobuddy' (threshold: {exit_threshold})")
            print(f"[wakeword] 🎯 Using openWakeWord with Speex noise suppression")
            print(f"[wakeword] 🔇 Running silently until detection...")

    start_t = time.time()
    audio_queue = []
    
    try:
        with sd.InputStream(**kwargs) as stream:
            while True:
                # Check timeout
                if timeout_sec and (time.time() - start_t) > timeout_sec:
                    raise TimeoutError("Wake word detection timeout")
                
                # Read audio chunk
                audio_chunk, overflowed = stream.read(blocksize)
                if overflowed:
                    print("[wakeword] ⚠️ Audio buffer overflow - some data lost")
                
                # Ensure we have int16 data
                if audio_chunk.dtype == np.float32:
                    audio_chunk = (audio_chunk * 32767).astype(np.int16)
                elif audio_chunk.dtype != np.int16:
                    audio_chunk = audio_chunk.astype(np.int16)
                
                # Flatten if stereo (shouldn't happen with channels=1)
                if len(audio_chunk.shape) > 1:
                    audio_chunk = audio_chunk[:, 0]
                
                # Get predictions for both models
                predictions = dual_model.predict(audio_chunk)
                entry_score = predictions.get('entry', 0.0)
                exit_score = predictions.get('exit', 0.0)
                
                # Check thresholds based on mode
                if conversation_mode:
                    # In conversation mode, only check for exit wake word
                    if exit_score >= exit_threshold:
                        print(f"\n[wakeword] 📤 EXIT WAKE WORD DETECTED! (score={exit_score:.3f})")
                        return 'exit', exit_score, Path(dual_model.exit_model_path).stem
                else:
                    # In wake mode, check for entry wake word
                    if entry_score >= entry_threshold:
                        print(f"\n[wakeword] 📥 ENTRY WAKE WORD DETECTED! (score={entry_score:.3f})")
                        return 'entry', entry_score, Path(dual_model.entry_model_path).stem
                
                # Optional: Show progress for debugging (can be removed for silent operation)
                if max(entry_score, exit_score) > max(entry_threshold, exit_threshold) * 0.3:
                    if conversation_mode:
                        bar = create_progress_bar(exit_score, exit_threshold)
                        print(f"\r[exit] {bar} {exit_score:.3f}", end="", flush=True)
                    else:
                        bar = create_progress_bar(max(entry_score, exit_score), max(entry_threshold, exit_threshold))
                        print(f"\r[wake] {bar} E:{entry_score:.3f} X:{exit_score:.3f}", end="", flush=True)
                
    except KeyboardInterrupt:
        print("\n[wakeword] Detection interrupted by user")
        raise
    except Exception as e:
        print(f"\n[wakeword] Detection error: {e}")
        raise