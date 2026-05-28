#!/usr/bin/env python3
"""Comprehensively patch sesame-finetune train.py to make W&B optional."""
import re

train_path = r"C:\Users\User\Desktop\israel-voice-workspace\sesame-finetune\train.py"

with open(train_path, "r") as f:
    content = f.read()

# Replace the entire main block to use conditional W&B
# Strategy: Add a wandb_run wrapper at the top and replace all wandb calls

# 1. After imports, add W&B dummy class
import_section_end = content.find("if __name__")
if import_section_end == -1:
    import_section_end = content.find("def parse_args")

wandb_wrapper = '''
# === W&B disabled wrapper ===
import os
if os.getenv("WANDB_MODE") == "disabled":
    class _WandbDummy:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
        def __bool__(self):
            return False
    wandb = _WandbDummy()
    wandb.run = None
    def wandb_log(*args, **kwargs): pass
    def wandb_save(*args, **kwargs): pass
else:
    # wandb already imported at top
    wandb_log = wandb.log
    wandb_save = wandb.save
# === end W&B wrapper ===

'''

content = content[:import_section_end] + wandb_wrapper + content[import_section_end:]
print("Added W&B dummy wrapper")

# 2. Replace wandb.log( with wandb_log(
content = re.sub(r'wandb\.log\(', 'wandb_log(', content)
print("Replaced wandb.log calls")

# 3. Replace wandb.save( with wandb_save(
content = re.sub(r'wandb\.save\(', 'wandb_save(', content)
print("Replaced wandb.save calls")

# 4. Replace assert wandb.run is not None
content = content.replace(
    'assert wandb.run is not None, "Wandb is not initialized"',
    'if wandb.run is None: print("Running without W&B")'
)
print("Fixed wandb.run assertion")

# 5. Replace wandb.Audio — return None
content = re.sub(r'wandb\.Audio\([^)]+\)', 'None', content)
print("Replaced wandb.Audio")

with open(train_path, "w") as f:
    f.write(content)

print(f"\nFully patched {train_path}")
print("Training will run without W&B when WANDB_MODE=disabled")
