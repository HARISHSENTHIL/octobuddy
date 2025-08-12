import sounddevice as sd

def list_input_devices():
    print("\n[devices] Available input devices:")
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_input_channels", 0) > 0:
            print(f"  #{i:2d}  {d['name']}  (in={d['max_input_channels']}, sr={int(d.get('default_samplerate', 0))})")
    print("[devices] Default input:", sd.default.device)

def main():
    list_input_devices()  # <-- print available input devices
    print("Octobuddy pipeline ready.")