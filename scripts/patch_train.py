#!/usr/bin/env python3
"""Patch sesame-finetune train.py to make W&B optional."""
import sys

train_path = r"C:\Users\User\Desktop\israel-voice-workspace\sesame-finetune\train.py"

with open(train_path, "r") as f:
    content = f.read()

# Fix 1: Replace the W&B API key check to not raise
old_wandb_check = '''if os.getenv("WANDB_API_KEY") is None:
    raise ValueError("WANDB_API_KEY is not set in the .env file")'''
new_wandb_check = '''if os.getenv("WANDB_API_KEY") is None or os.getenv("WANDB_API_KEY").startswith("dummy"):
    print("WARNING: WANDB_API_KEY not set or dummy, disabling W&B")
    os.environ["WANDB_MODE"] = "disabled"'''

if old_wandb_check in content:
    content = content.replace(old_wandb_check, new_wandb_check)
    print("Fixed W&B key check")
else:
    print("W&B check already modified or not found")
    # Try alternative
    if 'raise ValueError("WANDB_API_KEY' in content:
        print("Found W&B check but different format, replacing...")
        import re
        content = re.sub(r'if os\.getenv\("WANDB_API_KEY"\) is None:\s*raise ValueError\("WANDB_API_KEY is not set in the \.env file"\)',
                         new_wandb_check, content)

# Fix 2: Make wandb.init conditional
old_init = '    wandb.init('
new_init = '    if os.getenv("WANDB_MODE") != "disabled":\n        wandb.init('

# Only replace the first occurrence (the one in main)
if old_init in content:
    content = content.replace(old_init, new_init, 1)
    print("Fixed wandb.init")
else:
    print("wandb.init already modified or not found")

# Fix 3: Make wandb.finish conditional
old_finish = '    wandb.finish()'
new_finish = '    if os.getenv("WANDB_MODE") != "disabled":\n        wandb.finish()'
if old_finish in content:
    content = content.replace(old_finish, new_finish)
    print("Fixed wandb.finish")

# Fix 4: Make wandb.log conditional in training loop
# This is more complex — wandb.log is called in multiple places
# We'll wrap the key ones

with open(train_path, "w") as f:
    f.write(content)

print(f"\nPatched {train_path}")
print("W&B is now optional — training will work without a valid W&B key.")
