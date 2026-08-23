from simpok import get_device

found, name = get_device()
print(f"GPU: {found} ({name})")