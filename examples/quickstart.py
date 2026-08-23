from simpok import get_device

device = get_device()

if device.available:
    print(f"GPU: {device.name}")
else:
    print(f"No GPU: {device.reason}")
