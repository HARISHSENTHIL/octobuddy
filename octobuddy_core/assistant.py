# main.py
# Octobuddy: Wake -> VAD mic capture -> Whisper (EN) -> LLM (llama.cpp) -> Piper TTS
# - Lists input devices so you can pick the correct mic
# - Records ONLY when you speak; stops on trailing silence
# - Normalizes audio before Whisper
# - Filters bracketed/parenthetical non-speech
# - Short “intent” layer for greetings/story requests
# - Teacher-style answers with a tiny rolling memory

from __future__ import annotations
import json, time, subprocess, shlex, re, sys, os
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import sounddevice as sd
import soundfile as sf

# ---------- CONFIGURATION ----------
from dotenv import load_dotenv

# Load environment configuration
load_dotenv('config.env')

# Wake word settings
ENTRY_WAKEWORD_ONNX = os.getenv('WAKEWORD_ENTRY_MODEL', './models/Hey_octobuddy.onnx')
EXIT_WAKEWORD_ONNX = os.getenv('WAKEWORD_EXIT_MODEL', './models/bye_octa.onnx')
ENABLE_SPEEX = os.getenv('WAKEWORD_ENABLE_SPEEX', 'true').lower() == 'true'
ENTRY_WAKE_THRESHOLD = float(os.getenv('WAKEWORD_ENTRY_THRESHOLD', '0.020'))
EXIT_WAKE_THRESHOLD = float(os.getenv('WAKEWORD_EXIT_THRESHOLD', '0.025'))

# Whisper STT settings
WHISPER_BIN = os.getenv('WHISPER_BIN', '/home/hp/Desktop/whisper.cpp/build/bin/whisper-cli')
WHISPER_MODEL = os.getenv('WHISPER_MODEL', '/home/hp/Desktop/whisper.cpp/models/for-tests-ggml-base.en.bin')

# Piper TTS settings
PIPER_MODEL = os.getenv('PIPER_MODEL', './models/en_US-lessac-medium.onnx')
PIPER_CFG = os.getenv('PIPER_CONFIG', './models/en_US-lessac-medium.onnx.json')

# LLM settings
LLAMA_URL = os.getenv('LLM_URL', 'http://127.0.0.1:11434/api/generate')
LLAMA_MODEL = os.getenv('LLM_MODEL', 'llama3.2:3b')

# Audio settings
INPUT_DEVICE_INDEX = int(os.getenv('AUDIO_INPUT_DEVICE_INDEX', '5'))
SR = int(os.getenv('AUDIO_SAMPLE_RATE', '16000'))
BLOCK = int(os.getenv('AUDIO_BLOCKSIZE', '512'))

# Audio constants
MAX_RECORD_SEC = 30.0
COMMAND_TIMEOUT_SEC = 60.0
MAX_FOLLOW_UP_TURNS = 3

# ---- Behavior toggles ----
INACTIVITY_TIMEOUT_SEC = 60
POST_WAKE_DELAY_SEC    = 0.35
MUTE_AFTER_TTS_SEC     = 2.0


# Cache and safety settings
CACHE_PATH = Path(os.getenv('CACHE_FILE', 'cache.json'))
BLOCKED_TERMS_LIST = os.getenv('SAFETY_BLOCKED_TERMS', 'kill,violence,gun,suicide,sex').split(',')
BLOCKED_TERMS = set(term.strip() for term in BLOCKED_TERMS_LIST)
STOP_PHRASES  = {"bye octobuddy", "stop octobuddy", "goodbye octobuddy"}
SLEEP_PHRASES = {"sleep octobuddy", "go to sleep octobuddy"}

# Accept both [] and () tags like [background noise], (children talking), etc.
NON_SPEECH_REGEX = re.compile(
    r"""^\s*[\[\(]\s*(
        (background\s+)?noise|
        music|applause|silence|laughter|
        (children\s+talking)|inaudible|unclear|
        (speaking\s+in\s+foreign\s+language)|screaming
    )\s*[\]\)]\s*$""",
    re.IGNORECASE | re.VERBOSE
)

MUTE_WAKE_UNTIL = 0.0

# ----------------- Device helper -----------------
def list_input_devices():
    print("\n[devices] Available input devices:")
    try:
        devs = sd.query_devices()
    except Exception as e:
        print("[devices] Could not query devices:", e)
        return
    for i, d in enumerate(devs):
        if d.get("max_input_channels", 0) > 0:
            sr = d.get("default_samplerate", 0)
            # Highlight device that supports 16kHz (Whisper requirement)
            if sr == 16000:
                print(f"  #{i:2d}  {d['name']}  (in={d['max_input_channels']}, sr={int(sr)}) ✅ 16kHz - RECOMMENDED")
            else:
                print(f"  #{i:2d}  {d['name']}  (in={d['max_input_channels']}, sr={int(sr)})")
    print(f"[devices] Using device #{INPUT_DEVICE_INDEX} (configured in config.env)")
    print("[devices] Default (in,out):", sd.default.device)

# ----------------- Memory -----------------
class TopicMemory:
    def __init__(self, max_recent: int = 6):
        self.summary: str = ""
        self.recent: List[Tuple[str, str]] = []
        self.max_recent = max_recent
        self.turns_since_summarize = 0

    def add_turn(self, child: str, buddy: str):
        self.recent.append((child, buddy))
        if len(self.recent) > self.max_recent:
            self.recent = self.recent[-self.max_recent:]
        self.turns_since_summarize += 1

    def build_context_text(self) -> str:
        parts = []
        if self.summary:
            parts.append(f"Short summary so far: {self.summary}")
        if self.recent:
            rs = []
            for u, a in self.recent[-self.max_recent:]:
                rs.append(f"Child: {u}\nOctobuddy: {a}")
            parts.append("Recent turns:\n" + "\n".join(rs))
        return "\n\n".join(parts).strip()

    def maybe_update_summary(self, summarizer_fn):
        if self.turns_since_summarize < 3:
            return
        context = self.build_context_text()
        prompt = (
            "You are a concise assistant that writes short conversation summaries for a teacher.\n"
            "Summarize the child's topic, key facts learned, and the goal of the last exchanges in 2–3 short sentences.\n"
            "Use plain English, no lists, no quotes."
            "\n\nConversation:\n" + context + "\n\nSummary:"
        )
        new_summary = summarizer_fn(prompt).strip()
        if new_summary:
            self.summary = sanitize(new_summary, max_sentences=3, max_chars=300)
            self.recent = self.recent[-2:]
            self.turns_since_summarize = 0

# ---- tiny utils ----
def run(cmd: str, check: bool = False) -> subprocess.CompletedProcess:
    print("→", cmd)
    return subprocess.run(shlex.split(cmd), capture_output=True, text=True, check=check)

def speak_piper(text: str) -> None:
    global MUTE_WAKE_UNTIL
    outwav = Path("tts_out.wav").absolute()
    say = (text or "").replace("\n", " ").strip()[:220]
    if not say:
        return
    cmd = [
        sys.executable, "-m", "piper",
        "--model", PIPER_MODEL,
        "--config", PIPER_CFG,
        "--output_file", str(outwav)
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, text=True)
    p.communicate(say)
    
    # Ubuntu-compatible audio playback
    try:
        # Try aplay (ALSA) first
        subprocess.run(["aplay", str(outwav)], check=False)
    except FileNotFoundError:
        try:
            # Fallback to paplay (PulseAudio)
            subprocess.run(["paplay", str(outwav)], check=False)
        except FileNotFoundError:
            try:
                # Final fallback to ffplay
                subprocess.run(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(outwav)], check=False)
            except FileNotFoundError:
                print("[TTS] ❌ No audio playback method available")
    
    MUTE_WAKE_UNTIL = time.time() + MUTE_AFTER_TTS_SEC

def _rms(x: np.ndarray) -> float:
    x = x.astype(np.float64, copy=False)
    return float(np.sqrt(np.mean(x*x))) if x.size else 0.0

def _normalize_audio(y: np.ndarray, target_dbfs: float = -20.0) -> np.ndarray:
    eps = 1e-9
    rms = float(np.sqrt(np.mean(np.square(y))) + eps)
    current_db = 20.0 * np.log10(max(rms, eps))
    gain_db = target_dbfs - current_db
    gain = 10.0 ** (gain_db / 20.0)
    y = np.clip(y * gain, -1.0, 1.0)
    return y

def record_until_silence(path: str,
                         device_index: Optional[int] = None,
                         sr: int = SR,
                         block: int = BLOCK,
                         baseline_sec: float = 0.2,
                         start_gate_mul: float = 1.1,
                         start_min_ms: int = 60,
                         silence_gate_mul: float = 1.05,
                         end_sil_ms: int = 500,
                         max_record_sec: float = MAX_RECORD_SEC,
                         dual_model: Optional[DualWakeDetector] = None) -> Tuple[bool, bool]:
    """
    Simplified recording function - relies on openWakeWord's built-in Speex noise suppression
    Records until trailing silence, optimized for Ubuntu with minimal custom processing.
    
    Args:
        dual_model: Optional dual wake detector to check for exit wake words during recording
    
    Returns:
        Tuple[bool, bool]: (recording_success, exit_wake_detected)
    """
    kwargs = {"channels": 1, "samplerate": sr, "blocksize": block, "dtype": "float32"}
    if device_index is not None:
        kwargs["device"] = device_index

    print("[rec] Starting recording with openWakeWord noise suppression...")
    with sd.InputStream(**kwargs) as stream:
        # Quick baseline for basic energy detection
        acc = []
        t0 = time.time()
        while time.time() - t0 < baseline_sec:
            audio, _ = stream.read(block)
            acc.append(_rms(np.squeeze(audio)))
        baseline = float(np.median(acc)) if acc else 0.01
        start_gate = max(baseline * start_gate_mul, 0.008)
        sil_gate = max(baseline * silence_gate_mul, 0.006)

        print(f"[rec] Energy gates: start={start_gate:.4f}, silence={sil_gate:.4f}")

        # Wait for speech
        above_ms = 0.0
        wait_start = time.time()
        while time.time() - wait_start < 8.0:
            audio, _ = stream.read(block)
            e = _rms(np.squeeze(audio))
            if e >= start_gate:
                above_ms += (1000.0 * block / sr)
                if above_ms >= start_min_ms:
                    break
            else:
                above_ms = 0.0
            time.sleep(0.002)
        else:
            print("[rec] No speech detected.")
            return False, False

        # Record until trailing silence
        print("[rec] Recording...")
        frames = []
        trailing_sil_ms = 0.0
        rec_start = time.time()
        exit_detected = False
        
        while time.time() - rec_start < max_record_sec:
            audio, _ = stream.read(block)
            a = np.squeeze(audio).astype(np.float32, copy=False)
            frames.append(a)
            e = _rms(a)
            
            # Check for exit wake word during recording (if dual model provided)
            if dual_model and len(frames) >= (sr // block):  # Check every ~1 second of audio
                # Get the last second of audio for wake word detection
                recent_frames = frames[-(sr // block):]
                recent_audio = np.concatenate(recent_frames, axis=0)
                # Convert to int16 for wake word detection
                recent_audio_int16 = (recent_audio * 32767).astype(np.int16)
                
                if check_for_exit_wake_word(dual_model, recent_audio_int16):
                    exit_detected = True
                    print("\n[rec] 🚪 Exit wake word detected during recording - stopping")
                    break
            
            if e < sil_gate:
                trailing_sil_ms += (1000.0 * block / sr)
                if trailing_sil_ms >= end_sil_ms:
                    print("[rec] Stopping on silence.")
                    break
            else:
                trailing_sil_ms = 0.0
            time.sleep(0.002)

    if not frames:
        return False, exit_detected

    # Combine and save audio
    y = np.concatenate(frames, axis=0)
    y = _normalize_audio(y, target_dbfs=-20.0)
    sf.write(path, y, sr, subtype="PCM_16")
    
    try:
        size = os.path.getsize(path)
        print(f"[rec] Recorded: {size} bytes ({len(y)/sr:.1f}s)")
    except Exception:
        pass
    return True, exit_detected

def _read_whisper_txt(wav_path: str) -> str:
    txt_path = Path(wav_path + ".txt")
    if not txt_path.exists():
        return ""
    text = txt_path.read_text(encoding="utf-8").strip()
    # strip timestamps like [00:00.00 -> 00:01.23]
    text = re.sub(r"\[[0-9:.>\-\s]+\]\s*", "", text).strip()
    text = re.sub(r"\s+", " ", text)
    # drop pure non-speech tags like [background noise] or (children talking)
    if NON_SPEECH_REGEX.match(text):
        return ""
    # also drop if the whole line is a single parenthetical
    if re.fullmatch(r"\(\s*[^)]*\s*\)", text):
        return ""
    return text

def transcribe_whisper(wav: str) -> str:
    """
    Transcribe audio using Python Whisper (more reliable than whisper.cpp)
    Falls back to whisper.cpp if Python whisper fails
    """
    try:
        # Try Python Whisper with optimized settings for better recognition
        import whisper
        print("[whisper] 🗣️ Transcribing audio...")
        model = whisper.load_model("base.en")
        
        # Optimized transcription settings for better accuracy
        result = model.transcribe(
            wav, 
            language="en",
            word_timestamps=False,
            temperature=0.0,          # More deterministic
            best_of=2,               # Try 2 attempts for better accuracy
            beam_size=5,             # Better search
            condition_on_previous_text=False,  # Don't rely on previous context
            fp16=False,              # Use full precision
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6
        )
        
        text = result["text"].strip()
        if text and not text.lower().startswith(('[', '(', 'thank', 'thanks')):
            print(f"[whisper] ✅ Transcription: '{text}'")
            return text
        else:
            print(f"[whisper] ⚠️ Low quality transcription, trying fallback...")
    except Exception as e:
        print(f"[whisper] Python Whisper failed: {e}")
    
    # Fallback to whisper.cpp (original implementation)
    try:
        print("[whisper] Falling back to whisper.cpp...")
        # Relaxed first (no -su) to avoid early cutoffs
        cmd_plain = f'"{WHISPER_BIN}" -m "{WHISPER_MODEL}" -f "{wav}" -otxt -np'
        cp = run(cmd_plain)
        if cp.returncode == 0:
            text = _read_whisper_txt(wav)
            if text:
                return text
        # Fallback: single-utterance (fast end)
        cmd_su = f'"{WHISPER_BIN}" -m "{WHISPER_MODEL}" -f "{wav}" -otxt -np -su'
        cp = run(cmd_su)
        if cp.returncode == 0:
            return _read_whisper_txt(wav)
    except Exception as e:
        print(f"[whisper] whisper.cpp also failed: {e}")
    
    return ""

def ask_llama_raw(prompt: str, n_predict: int = 200, temperature: float = 0.7) -> str:
    """
    Use Ollama's /api/generate endpoint (non-streaming) with stop tokens.
    Make sure `ollama serve` is running on localhost:11434.
    """
    payload = {
        "model": LLAMA_MODEL,      # Use configured model
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": n_predict,
            "num_ctx": 4096,  # context window
            "stop": ["\n", "Child:", "Octobuddy:"]
        }
    }
    try:
        cp = subprocess.run(
            ["curl", "-s", LLAMA_URL, "-d", json.dumps(payload)],
            capture_output=True, text=True
        )
        if cp.returncode != 0 or not cp.stdout.strip():
            return ""
        data = json.loads(cp.stdout)
        # non-streaming response puts text in "response"
        txt = (data or {}).get("response", "")
        return str(txt).strip()
    except Exception:
        return ""


def ask_llama_teacher(child_text: str, memory_text: str) -> str:
    context_block = f"\n\nContext (short):\n{memory_text}" if memory_text else ""
    prompt = (
        "You are Octobuddy, a warm, kid-safe teacher.\n"
        "Use the context to keep the same topic if the child asks a follow-up like 'why?' or 'then what?'.\n"
        "Explain clearly in 2–3 short sentences with simple words and one tiny example or analogy.\n"
        "End with a gentle follow-up question that checks understanding.\n"
        "Do NOT reply with one-word or two-word answers like 'Yes', 'No', or 'Sure'.\n"
        "Avoid lists or roleplay; speak directly to the child."
        f"{context_block}\n\n"
        f"Child: {child_text}\nOctobuddy:"
    )
    raw = ask_llama_raw(prompt, n_predict=240, temperature=0.7)
    return raw or "Let's try another question! What part should we explore first?"

def summarize_with_llm(prompt: str) -> str:
    return ask_llama_raw(prompt, n_predict=120, temperature=0.4)

def quick_intent_reply(text: str) -> Optional[str]:
    t = text.lower().strip()
    if re.search(r"\b(tell|give)\b.*\b(story)\b", t) or t in {"story", "a story", "give me a story"}:
        return ("Once there was a tiny seed that thought it was too small to matter. "
                "Each day it drank a little water and sunlight, and kept trying even when the wind was cold. "
                "Slowly it became a bright flower that cheered the whole garden. "
                "What do you think helped the seed the most?")
    if t in {"hi", "hello", "hey"} or "how are you" in t or "good morning" in t:
        return ("Hi! I’m glad you’re here. What would you like to learn about today?")
    if "thank you" in t or "thanks" in t:
        return ("You’re welcome! Want to try another question or a short story?")
    return None

def safe(text: str) -> bool:
    t = text.lower()
    return not any(bad in t for bad in BLOCKED_TERMS)

def cache_get(key: str) -> Optional[str]:
    if not CACHE_PATH.exists():
        return None
    try:
        data = json.loads(CACHE_PATH.read_text())
        return data.get(key)
    except Exception:
        return None

def cache_put(key: str, value: str):
    data = {}
    if CACHE_PATH.exists():
        try:
            data = json.loads(CACHE_PATH.read_text())
        except Exception:
            data = {}
    data[key] = value
    CACHE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2))

def sanitize(reply: str, max_sentences: int = 3, max_chars: int = 420) -> str:
    reply = re.split(r"\b(?:Child|Octobuddy)\s*:\s*", reply)[0]
    reply = re.sub(r"\([^)]*\)", "", reply).strip().replace("\n", " ")
    sentences = re.split(r"(?<=[.!?])\s+", reply)
    reply = " ".join(s for s in sentences[:max_sentences]).strip()
    if len(reply) > max_chars:
        reply = reply[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;")
    return reply or "Let me explain in a simple way."

def norm_text(t: str) -> str:
    t = re.sub(r"[^a-z\s]", " ", t.lower())
    return re.sub(r"\s+", " ", t).strip()

def is_stop_phrase(text: str) -> bool:
    t = norm_text(text)
    return any(phrase in t for phrase in STOP_PHRASES)

def is_sleep_phrase(text: str) -> bool:
    t = norm_text(text)
    return any(phrase in t for phrase in SLEEP_PHRASES)

# ---- wake-word (Ubuntu-optimized with openWakeWord) ----
USE_WAKE = True
oww_model = None
try:
    from wake_word import DualWakeDetector, wait_for_dual_wake
    oww_model = DualWakeDetector(ENTRY_WAKEWORD_ONNX, EXIT_WAKEWORD_ONNX, enable_speex=ENABLE_SPEEX)
    print(f"[wakeword] ✅ Dual wake word detector loaded!")
    print(f"[wakeword] 🎯 Using openWakeWord with Speex noise suppression: {ENABLE_SPEEX}")
    print(f"[wakeword] 💡 Will run silently until wake word is detected")
except Exception as e:
    print(f"[wakeword] ❌ Disabled (reason: {e}) → using push-to-talk mode")
    print(f"[wakeword] 💡 Press ENTER when you want to speak")
    USE_WAKE = False

def check_for_exit_wake_word(dual_model: DualWakeDetector, audio_chunk: np.ndarray) -> bool:
    """
    Check audio chunk for exit wake word without creating a separate audio stream
    Returns True if exit wake word is detected
    """
    try:
        # Get predictions for the audio chunk
        predictions = dual_model.predict(audio_chunk)
        exit_score = predictions.get('exit', 0.0)
        
        # Check if exit threshold is met
        if exit_score >= EXIT_WAKE_THRESHOLD:
            exit_name = Path(dual_model.exit_model_path).stem
            print(f"\n[wakeword] 🚪 EXIT WAKE WORD DETECTED! '{exit_name}' (score={exit_score:.3f})")
            return True
        
        # Optional: Show progress for debugging (can be removed for silent operation)
        if exit_score > EXIT_WAKE_THRESHOLD * 0.3:
            from octobuddy_core.wake_word import create_progress_bar
            bar = create_progress_bar(exit_score, EXIT_WAKE_THRESHOLD)
            print(f"\r[exit] {bar} {exit_score:.3f}", end="", flush=True)
        
        return False
    except Exception as e:
        print(f"[wakeword] ❌ Exit wake word check error: {e}")
        return False

def main():
    list_input_devices()  # check list; set INPUT_DEVICE_INDEX accordingly
    print("Octobuddy Educational Assistant ready.")
    listening_for_wake = True
    conversation_turns = 0  # Track follow-up turns
    last_activity = time.time()
    memory = TopicMemory(max_recent=6)

    while True:
        try:
            # ---- Wake phase ----
            if listening_for_wake and USE_WAKE and oww_model is not None:
                conversation_turns = 0  # Reset conversation counter
                while time.time() < MUTE_WAKE_UNTIL:
                    time.sleep(0.05)
                try:
                    wake_type, score, name = wait_for_dual_wake(
                        oww_model,
                        entry_threshold=ENTRY_WAKE_THRESHOLD,
                        exit_threshold=EXIT_WAKE_THRESHOLD,
                        input_device_index=INPUT_DEVICE_INDEX,
                        timeout_sec=300,                # 5 minute timeout
                        print_startup=listening_for_wake,  # Only print on first startup
                        conversation_mode=False  # Looking for entry wake word
                    )
                    if wake_type == 'entry':
                        print(f"[wakeword] ✨ WAKE WORD DETECTED! '{name}' (score={score:.2f})")
                        print(f"[wakeword] 🎤 Ready! Please ask your question now...")
                        
                        # Clear countdown to indicate when to start speaking
                        for i in range(3, 0, -1):
                            print(f"[wakeword] 🕐 Starting in {i}...", end='\r')
                            time.sleep(1)
                        print(f"[wakeword] 🎤 SPEAK NOW! I'm listening...         ")
                        
                        listening_for_wake = False
                        last_activity = time.time()
                        print("[wakeword] 💡 Say 'Bye Octobuddy' anytime to end the conversation")
                    else:
                        # Unexpected exit wake word while in wake mode
                        print(f"[wakeword] ⚠️ Exit wake word detected while sleeping - ignoring")
                        continue
                except TimeoutError:
                    continue
            elif listening_for_wake:
                input("Press ENTER to talk…")
                listening_for_wake = False
                last_activity = time.time()

            # ---- Command listening phase ----
            # Check if we've been waiting too long for a command after wake word
            if time.time() - last_activity > COMMAND_TIMEOUT_SEC:
                print("[timeout] ⏰ No command received, going back to sleep...")
                listening_for_wake = True
                continue

            wav = "utterance.wav"
            print(f"[recording] 🎤 Recording your question...")
            print(f"[recording] 💡 Speak clearly and naturally - I'll stop when you're done")
            
            # More sensitive recording parameters after wake word
            ok, exit_detected = record_until_silence(
                wav, device_index=INPUT_DEVICE_INDEX,
                sr=SR, block=BLOCK, 
                baseline_sec=0.2,           # Even shorter calibration
                start_gate_mul=1.1,         # Very sensitive (easier to trigger)
                start_min_ms=60,            # Very short minimum speech
                silence_gate_mul=1.05,      # Very sensitive to silence
                end_sil_ms=500,             # Quick silence detection  
                max_record_sec=MAX_RECORD_SEC,
                dual_model=oww_model if USE_WAKE else None  # Pass dual model for exit detection
            )
            
            # Check if exit wake word was detected during recording
            if exit_detected:
                print("[wakeword] 👋 Exit wake word detected during recording - goodbye!")
                speak_piper("Goodbye! Say 'Hey Octobuddy' if you want to chat again.")
                listening_for_wake = True
                continue
            
            if not ok:
                # no speech detected within timeout
                print("[speech] 🔇 I didn't hear anything. Try saying 'Hey Octobuddy' again!")
                print("[speech] 💡 Tip: Speak clearly right after you hear the wake word confirmation")
                listening_for_wake = True
                continue

            # Check recorded file
            import os
            if os.path.exists(wav):
                size = os.path.getsize(wav)
                print(f"[recording] ✅ Audio recorded successfully: {size} bytes")
            else:
                print("[recording] ❌ Audio file was not created!")
                listening_for_wake = True
                continue

            print("[transcription] 🗣️ Converting speech to text...")
            text = transcribe_whisper(wav).strip()
            print(f"[transcription] Result: '{text}'")
            
            if not text:
                print("[transcription] ❌ No text detected from audio")
                speak_piper("I didn't catch that. Please say 'Hey Octobuddy' to wake me up again.")
                listening_for_wake = True
                continue
            else:
                print(f"[transcription] ✅ Detected: '{text}'")

            last_activity = time.time()
            print(f"[child] {text}")

            # commands
            if is_stop_phrase(text) or is_sleep_phrase(text):
                speak_piper("Okay, I'm going to sleep. Say 'Hey Octobuddy' to wake me up again!")
                listening_for_wake = True
                time.sleep(0.4)
                continue

            if not safe(text):
                speak_piper("Let's talk about something else. What would you like to learn about?")
                # Don't return to wake mode for safety - allow them to ask another question
                last_activity = time.time()
                continue

            # quick intents for short, common asks
            intent = quick_intent_reply(text)
            if intent:
                reply = intent
            else:
                memory_text = memory.build_context_text()
                raw = ask_llama_teacher(text, memory_text)
                reply = sanitize(raw)

            if safe(reply):
                cache_put(text.lower(), reply)

            print(f"[octobuddy] {reply}")
            speak_piper(reply)

            memory.add_turn(text, reply)
            memory.maybe_update_summary(lambda p: ask_llama_raw(p, n_predict=120, temperature=0.4))
            
            # Handle follow-up conversation logic
            conversation_turns += 1
            last_activity = time.time()
            
            # Check if we should return to wake word mode
            if conversation_turns >= MAX_FOLLOW_UP_TURNS:
                print(f"[conversation] 😴 Max turns reached ({MAX_FOLLOW_UP_TURNS}), returning to sleep mode...")
                speak_piper("That was a great conversation! Say 'Hey Octobuddy' if you have more questions.")
                listening_for_wake = True
                time.sleep(1.0)
            else:
                print(f"[conversation] 🔄 Turn {conversation_turns}/{MAX_FOLLOW_UP_TURNS} - listening for follow-up question...")
                print(f"[conversation] 💡 Say 'Bye Octobuddy' to end the conversation early")
                # Continue listening for follow-up questions with a shorter timeout
                last_activity = time.time()

        except KeyboardInterrupt:
            print("\nBye!")
            break
        except Exception as e:
            print("[error]", e)
            try:
                speak_piper("Sorry, something went wrong.")
            except Exception:
                pass
            time.sleep(0.5)

if __name__ == "__main__":
    main()