#!/usr/bin/env python3
"""Phase 3: CSM-1B training orchestrator for sesame-finetune."""
import os
import sys
import json
import subprocess
import argparse
from pathlib import Path

WORKSPACE = "C:/Users/User/Desktop/israel-voice-workspace"
SESAME_REPO = os.path.join(WORKSPACE, "sesame-finetune")
META_DIR = os.path.join(WORKSPACE, "voice_data", "israel", "metadata")
DATA_DIR = os.path.join(SESAME_REPO, "data")
EXP_DIR = os.path.join(SESAME_REPO, "exp")

def run_cmd(cmd, desc=None):
    if desc:
        print("\n" + "="*60)
        print(desc)
        print("="*60)
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    return result.returncode

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=WORKSPACE)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--phase", choices=["pretokenize", "train", "all"], default="all")
    args = parser.parse_args()

    global WORKSPACE, SESAME_REPO, META_DIR, DATA_DIR, EXP_DIR
    WORKSPACE = args.workspace
    SESAME_REPO = os.path.join(WORKSPACE, "sesame-finetune")
    META_DIR = os.path.join(WORKSPACE, "voice_data", "israel", "metadata")
    DATA_DIR = os.path.join(SESAME_REPO, "data")
    EXP_DIR = os.path.join(SESAME_REPO, "exp")

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(EXP_DIR, exist_ok=True)

    train_json = os.path.join(META_DIR, "train.json")
    val_json = os.path.join(META_DIR, "val.json")

    if not os.path.exists(train_json):
        print(f"ERROR: {train_json} not found!")
        sys.exit(1)

    with open(train_json) as f:
        train_data = json.load(f)
    print(f"Train samples: {len(train_data)}")

    if args.phase in ("pretokenize", "all"):
        hdf5_path = os.path.join(DATA_DIR, "tokens.hdf5")
        cmd = (f'cd {SESAME_REPO} && python pretokenize.py '
               f'--train_data "{train_json}" '
               f'--val_data "{val_json}" '
               f'--output "{hdf5_path}" '
               f'--device cuda')
        rc = run_cmd(cmd, "Step 1: Pre-tokenizing with Mimi codec")
        if rc != 0:
            print("ERROR: pretokenize failed!")
            sys.exit(1)

    if args.phase in ("train", "all"):
        hdf5_path = os.path.join(DATA_DIR, "tokens.hdf5")
        config_path = os.path.join(SESAME_REPO, "configs", "finetune_3070.yaml")
        if not os.path.exists(hdf5_path):
            print(f"ERROR: {hdf5_path} not found!")
            sys.exit(1)
        cmd = (f'cd {SESAME_REPO} && python train.py '
               f'--data "{hdf5_path}" '
               f'--config "{config_path}" '
               f'--output_dir "{EXP_DIR}" '
               f'--n_epochs {args.epochs} '
               f'--use_amp '
               f'--log_every 10 '
               f'--val_every 100 '
               f'--save_every 500 '
               f'--gen_every 500 '
               f'--gen_sentences "The voice of the Chairman speaks with clarity and authority."')
        rc = run_cmd(cmd, f"Step 2: Training CSM-1B ({args.epochs} epochs)")
        if rc == 0:
            print(f"\nTraining complete! Model saved to: {EXP_DIR}")
        else:
            print("Training completed with errors. Check logs.")

if __name__ == "__main__":
    main()
