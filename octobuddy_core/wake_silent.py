#!/usr/bin/env python3
"""
Silent Wake Word Detection
Only activates and prints when wake word is actually detected
No continuous loop output - truly silent operation
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from collections import deque
import time
import numpy as np
import sounddevice as sd
import onnxruntime as ort
import librosa

def _rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    x = x.astype(np.float64, copy=False)
    return float(np.sqrt(np.mean(np.square(x))))

def create_progress_bar(score: float, threshold: float, complete: bool = False) -> str:
    """Create a visual progress bar for wake word detection"""
    if complete:
        return "█████████████████████ ✅"
    
    # Calculate progress (0 to 1)
    max_display = threshold * 3  # Show progress up to 3x threshold
    progress = min(score / max_display, 1.0)
    
    # Create bar
    filled = int(progress * 20)
    empty = 20 - filled
    
    bar = "█" * filled + "░" * empty
    
    # Add indicator when approaching threshold
    if score >= threshold * 0.8:
        indicator = " 🔥"
    elif score >= threshold * 0.5:
        indicator = " 🟡"
    else:
        indicator = " 🔵"
    
    return bar + indicator

def _calibrate_noise(
    seconds: float = 1.0,
    samplerate: int = 16000,
    input_device_index: Optional[int] = None,
    blocksize: int = 1024,
) -> float:
    """Silent microphone calibration"""
    kwargs = {
        "channels": 1,
        "samplerate": samplerate,
        "blocksize": blocksize,
        "dtype": "float32",
    }
    if input_device_index is not None:
        kwargs["device"] = input_device_index

    acc: List[float] = []
    with sd.InputStream(**kwargs) as stream:
        t_end = time.time() + seconds
        while time.time() < t_end:
            audio, _ = stream.read(blocksize)
            acc.append(_rms(np.squeeze(audio)))
    return float(np.median(acc)) if acc else 0.0

class SilentWakeDetector:
    """
    Truly silent wake word detector - no continuous output
    Only prints when wake word is detected or on startup
    """
    def __init__(self, onnx_path: str):
        onnx_path = str(Path(onnx_path))
        self.sess = ort.InferenceSession(
            onnx_path,
            providers=[
                ("CoreMLExecutionProvider", {"MLComputeUnits":"ALL","ModelFormat":"MLProgram"}), 
                "CPUExecutionProvider"
            ]
        )
        inp = self.sess.get_inputs()[0]
        if not (isinstance(inp.shape, list) and len(inp.shape) == 3 and inp.shape[2] == 96):
            raise ValueError(f"Model must take [1,N,96] format, got {inp.shape}")
        
        self.inp_name = inp.name
        self.n_mels = inp.shape[1]

    def _logmel(self, wav_16k: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Create log mel spectrogram"""
        S = librosa.feature.melspectrogram(
            y=wav_16k.astype(np.float32), sr=sr,
            n_fft=512, hop_length=160, win_length=400,
            n_mels=self.n_mels, fmin=20, fmax=8000, power=2.0
        )
        return np.log10(np.maximum(S, 1e-10))

    def predict(self, audio_1s: np.ndarray) -> dict:
        """Predict wake word score silently"""
        M = self._logmel(audio_1s, 16000)
        T = M.shape[1]
        if T < 96:
            pad = np.zeros((self.n_mels, 96 - T), dtype=M.dtype)
            M = np.concatenate([M, pad], axis=1)
        else:
            M = M[:, -96:]
        X = M[np.newaxis, :, :].astype(np.float32)
        out = self.sess.run(None, {self.inp_name: X})

        # Extract score
        score = None
        for o in out:
            if o.ndim == 2 and o.shape[-1] == 1:
                score = float(o[0, 0]); break
            if o.ndim == 2 and o.shape[-1] == 2:
                score = float(o[0, 1]); break
            if o.size > 0:
                score = float(o.flat[-1]); break
        if score is None:
            score = 0.0
        return {"hey_octobuddy": score}

def wait_for_wake_silent(
    model: SilentWakeDetector,
    threshold: float = 0.005,
    samplerate: int = 16000,
    input_device_index: Optional[int] = None,
    timeout_sec: Optional[float] = None,
    print_startup: bool = True,
    **kwargs  # Accept additional arguments for compatibility
) -> Tuple[str, float]:
    """
    Silent wake word detection - only prints on detection or startup
    No continuous loop output - efficient and quiet operation
    """
    
    # Setup audio streaming
    chunk_ms = 750  # Longer chunks for efficiency (750ms)
    blocksize = int(samplerate * (chunk_ms / 1000.0))
    blocksize = max(blocksize, 512)
    win_samples = int(samplerate * 1.0)  # 1 second window
    ring = np.zeros(win_samples, dtype=np.float32)

    kwargs = {
        "channels": 1,
        "samplerate": samplerate,
        "blocksize": blocksize,
        "dtype": "float32",
    }
    if input_device_index is not None:
        kwargs["device"] = input_device_index

    # Silent calibration
    if print_startup:
        print("[wakeword] 🔧 Initializing wake word detection...")
    
    baseline = _calibrate_noise(
        seconds=0.5,  # Quick calibration
        samplerate=samplerate,
        input_device_index=input_device_index,
        blocksize=1024,
    )
    energy_gate = baseline * 1.8  # Higher gate for efficiency
    
    if print_startup:
        print(f"[wakeword] 👂 Listening for 'Hey Octobuddy' (threshold: {threshold})")
        print(f"[wakeword] 📊 Progress bar will show detection strength")
        print(f"[wakeword] 🔇 Running silently until wake word detected...")

    start_t = time.time()
    last_fire = 0.0

    # Scoring and detection
    score_buffer = deque(maxlen=2)  # Smaller buffer for faster response
    consecutive_detections = 0
    required_consecutive = 1  # Only need 1 detection for responsiveness

    with sd.InputStream(**kwargs) as stream:
        while True:
            if timeout_sec is not None and (time.time() - start_t) > timeout_sec:
                raise TimeoutError("wake-word wait timed out")

            # Read audio chunk
            audio, _ = stream.read(blocksize)
            audio = np.squeeze(audio).astype(np.float32, copy=False)

            # Update ring buffer
            n = audio.size
            if n >= win_samples:
                ring[:] = audio[-win_samples:]
            else:
                ring = np.roll(ring, -n)
                ring[-n:] = audio

            # Energy gate - skip if too quiet (silent operation)
            e = _rms(audio)
            current_time = time.time()
            
            if e < energy_gate:
                consecutive_detections = 0
                time.sleep(0.2)  # Longer sleep when quiet for efficiency
                continue

            # Predict wake word score
            preds: Dict[str, float] = model.predict(ring)
            score = float(preds.get("hey_octobuddy", 0.0))
            score_buffer.append(score)
            smoothed = float(np.mean(score_buffer)) if len(score_buffer) > 0 else score

            # Detection logic with visual progress bar
            if smoothed >= threshold and (current_time - last_fire) >= 1.0:  # 1 second cooldown
                consecutive_detections += 1
                
                # Show progress bar during detection
                if consecutive_detections == 1:
                    progress_bar = create_progress_bar(smoothed, threshold)
                    print(f"[wakeword] 🔥 Detecting... {progress_bar} {smoothed:.3f}")
                
                if consecutive_detections >= required_consecutive:
                    final_bar = create_progress_bar(smoothed, threshold, complete=True)
                    print(f"[wakeword] ✅ CONFIRMED! {final_bar} {smoothed:.3f}")
                    last_fire = current_time
                    return "hey_octobuddy", smoothed
                    
                time.sleep(0.05)  # Brief pause during detection
            else:
                consecutive_detections = 0
                
                # Show occasional progress when speaking but not detecting
                if e > energy_gate and smoothed > 0.002:
                    progress_bar = create_progress_bar(smoothed, threshold)
                    print(f"[wakeword] 👂 Listening... {progress_bar} {smoothed:.3f}", end='\r')

            # Efficient sleep to prevent tight loop
            time.sleep(0.05)

# Alias to maintain compatibility
DirectWakeONNX = SilentWakeDetector
wait_for_wake = wait_for_wake_silent
