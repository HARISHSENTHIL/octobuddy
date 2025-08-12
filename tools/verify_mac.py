import onnxruntime as ort

print("ONNX Runtime version:", ort.__version__)
print("Available providers:", ort.get_available_providers())

if "CoreMLExecutionProvider" in ort.get_available_providers():
    print("✅ Core ML EP is available — ANE/GPU ready")
else:
    print("⚠️ Core ML EP not found — running CPU only")
