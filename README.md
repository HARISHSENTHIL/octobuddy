# 🐙 OctoBuddy - Educational Voice Assistant

A sophisticated voice-activated AI assistant designed specifically for children's education.

## 🎯 Features

- **Silent Wake Word Detection**: "Hey Octobuddy" with visual progress bar
- **Educational Responses**: Age-appropriate, teacher-style answers
- **Multi-turn Conversations**: Follow-up questions and context awareness
- **Safety First**: Content filtering and child-safe interactions
- **High-quality TTS**: Natural voice synthesis with Piper
- **Optimized Performance**: Efficient, event-driven operation

## 📁 Project Structure

```
octobuddy/
├── 🚀 run_octobuddy.py          # Main launcher script
├── 📦 octobuddy_core/           # Core application
│   ├── main.py                  # Main application logic
│   └── wake_silent.py           # Wake word detection
├── 🤖 models/                   # AI models
│   ├── Hey_octobuddy.onnx       # Wake word model
│   ├── en_US-lessac-medium.onnx # TTS voice model
│   └── en_US-lessac-medium.onnx.json
├── 🛠️  tools/                   # Utilities and testing
│   ├── find_optimal_threshold.py # Optimize wake word sensitivity
│   ├── mic_selector.py          # Choose best microphone
│   ├── test_improved_ux.py      # Test UX improvements
│   └── run_tests.py             # Test suite runner
├── 📚 docs/                     # Documentation and config
│   ├── config.yaml              # Configuration file
│   └── requirements.txt         # Python dependencies
├── 🎵 audio/                    # Audio files and cache
│   ├── tts_out.wav              # TTS output (temporary)
│   ├── utterance.wav            # Voice input (temporary)
│   └── cache.json               # Response cache
└── 🐍 venv/                     # Python virtual environment
```

## 🚀 Quick Start

### 1. Setup
```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies (if needed)
pip install -r docs/requirements.txt
```

### 2. Run OctoBuddy
```bash
# Simple launch
python run_octobuddy.py

# Or run directly
cd octobuddy_core && python main.py
```

### 3. Optimize (First Time)
```bash
# Find optimal wake word threshold for your voice
python tools/find_optimal_threshold.py

# Test microphone selection
python tools/mic_selector.py
```

## 🎯 Usage

1. **Start**: Run `python run_octobuddy.py`
2. **Wake**: Say "Hey Octobuddy" clearly
3. **Wait**: Look for the countdown: 3... 2... 1... SPEAK NOW!
4. **Ask**: Ask educational questions
5. **Follow-up**: Ask up to 3 follow-up questions
6. **Sleep**: System returns to listening mode

## 🛠️ Tools

- `tools/find_optimal_threshold.py` - Optimize wake word sensitivity
- `tools/mic_selector.py` - Choose and test microphones
- `tools/test_improved_ux.py` - Test user experience features
- `tools/run_tests.py` - Complete test suite

## 🔧 Configuration

Edit `docs/config.yaml` to customize:
- Wake word sensitivity
- Audio settings
- LLM parameters
- Safety filters

## 📋 Requirements

- Python 3.10+
- Ollama with llama3.1:8b model
- Compatible microphone
- macOS or Linux

## 🎓 Educational Focus

OctoBuddy is designed as an educational toy that:
- Provides age-appropriate explanations
- Encourages curiosity and learning
- Uses simple vocabulary with examples
- Asks follow-up questions to check understanding
- Maintains conversation context

Perfect for children's STEM education and interactive learning!
