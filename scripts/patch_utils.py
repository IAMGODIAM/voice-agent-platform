#!/usr/bin/env python3
"""Patch utils.py to make silentcipher/watermarker optional."""
import re

utils_path = r"C:\Users\User\Desktop\israel-voice-workspace\sesame-finetune\utils.py"

with open(utils_path, "r") as f:
    content = f.read()

# Replace the generator import to handle missing silentcipher
old_import = "from generator import Generator, load_llama3_tokenizer, load_watermarker"
new_import = """try:
    from generator import Generator, load_llama3_tokenizer, load_watermarker
except ImportError as e:
    print(f"WARNING: Could not import from generator: {e}")
    # Fallback: define dummy functions
    def load_watermarker(device=None):
        return None
    class Generator:
        def __init__(self, *args, **kwargs):
            pass
        def generate(self, *args, **kwargs):
            return None
    from generator import load_llama3_tokenizer"""

content = content.replace(old_import, new_import)

# Also fix the load_watermarker call in generate_audio
old_watermarker = """    watermarker = load_watermarker(device=device)"""
new_watermarker = """    try:
        watermarker = load_watermarker(device=device)
    except Exception as e:
        print(f"WARNING: load_watermarker failed: {e}, running without watermarking")
        watermarker = None"""

if old_watermarker in content:
    content = content.replace(old_watermarker, new_watermarker)
    print("Fixed load_watermarker in generate_audio")

with open(utils_path, "w") as f:
    f.write(content)

print(f"Patched {utils_path}")
print("Watermarker is now optional")
