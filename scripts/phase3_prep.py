"""
Phase 3 Preparation: Create .env, VRAM-optimized config, and launch training.
"""
import os
import shutil

WORKSPACE = "C:/Users/User/Desktop/israel-voice-workspace"
SESAME_REPO = os.path.join(WORKSPACE, "sesame-finetune")
CSM_REPO = os.path.join(WORKSPACE, "csm")

# 1. Create .env file
env_content = f"""CSM_REPO_PATH={CSM_REPO}
MIMI_SAMPLE_RATE=24000
BACKBONE_FLAVOR=llama-1B
DECODER_FLAVOR=llama-100M
TEXT_VOCAB_SIZE=128256
AUDIO_VOCAB_SIZE=2051
AUDIO_NUM_CODEBOOKS=32
"""
with open(os.path.join(SESAME_REPO, ".env"), "w") as f:
    f.write(env_content)
print("Created .env")

# 2. Create VRAM-optimized config for RTX 3070 8GB
# batch_size=4, grad_acc_steps=2 gives effective batch of 8
# Smaller batch to fit in 8GB VRAM
config = {
    "batch_size": 4,
    "grad_acc_steps": 2,
    "learning_rate": 3e-5,
    "max_grad_norm": 1.3,
    "warmup_steps": 100,
    "weight_decay": 0.002,
    "lr_decay": "linear",
    "decoder_loss_weight": 0.5,
}

import yaml
config_path = os.path.join(SESAME_REPO, "configs", "finetune_3070.yaml")
with open(config_path, "w") as f:
    yaml.dump(config, f, default_flow_style=False)
print(f"Created VRAM-optimized config: {config_path}")
print(f"Config: {config}")

# 3. Create data directory
os.makedirs(os.path.join(SESAME_REPO, "data"), exist_ok=True)

# 4. Create exp directory
os.makedirs(os.path.join(SESAME_REPO, "exp"), exist_ok=True)

print("\nPhase 3 preparation complete!")
print(f"  .env: {os.path.join(SESAME_REPO, '.env')}")
print(f"  config: {config_path}")
print(f"\nTo run training:")
print(f"  python pretokenize.py --train_data <train.json> --val_data <val.json> --output ./data/tokens.hdf5")
print(f"  python train.py --data ./data/tokens.hdf5 --config ./configs/finetune_3070.yaml --n_epochs 25 --use_amp")
