#!/usr/bin/env python3
"""
Phase 3: CSM-1B Fine-Tuning using sesame-finetune (knottwill)
Uses the proven sesame-finetune pipeline with Mimi/Llama tokenizers.
"""
import os
import sys
import json
import subprocess
import argparse

WORKSPACE = "C:/Users/User/Desktop/israel-voice-workspace"
META_DIR = os.path.join(WORKSPACE, "voice_data", "israel", "metadata")
SESAME_REPO = os.path.join(WORKSPACE, "sesame-finetune")
CSM_REPO = os.path.join(WORKSPACE, "csm")

def run_cmd(cmd, desc=None):
    if desc:
        print("\n" + "="*60)
        print(desc)
        print("="*60)
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=False)
    return result.returncode

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=WORKSPACE)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--skip-pretokenize", action="store_true")
    args = parser.parse_args()

    global WORKSPACE, META_DIR, SESAME_REPO, CSM_REPO
    WORKSPACE = args.workspace
    META_DIR = os.path.join(WORKSPACE, "voice_data", "israel", "metadata")
    SESAME_REPO = os.path.join(WORKSPACE, "sesame-finetune")
    CSM_REPO = os.path.join(WORKSPACE, "csm")

    os.chdir(WORKSPACE)

    print("="*60)
    print("Phase 3: CSM-1B Fine-Tuning via sesame-finetune")
    print(f"Workspace: {WORKSPACE}")
    print(f"Epochs: {args.epochs}")
    print("="*60)

    train_json = os.path.join(META_DIR, "train.json")
    val_json = os.path.join(META_DIR, "val.json")

    if not os.path.exists(train_json):
        print(f"ERROR: {train_json} not found!")
        sys.exit(1)

    with open(train_json) as f:
        train_data = json.load(f)
    with open(val_json) as f:
        val_data = json.load(f)

    print(f"\nDataset: {len(train_data)} train, {len(val_data)} val samples")

    if not args.skip_pretokenize:
        pkl_path = os.path.join(WORKSPACE, "tokenized_data.pkl")
        cmd = (f'python "{os.path.join(SESAME_REPO, "pretokenize.py")}" '
               f'--train_data "{train_json}" '
               f'--val_data "{val_json}" '
               f'--output "{pkl_path}"')
        run_cmd(cmd, "Step 1: Pre-tokenizing with Mimi codec")
    else:
        pkl_path = os.path.join(WORKSPACE, "tokenized_data.pkl")

    if os.path.exists(pkl_path):
        gen_text = "The voice of the Chairman speaks with clarity and authority."
        cmd = (f'python "{os.path.join(SESAME_REPO, "finetune.py")}" '
               f'--data "{pkl_path}" '
               f'--config "{os.path.join(SESAME_REPO, "configs", "default.yaml")}" '
               f'--n_epochs {args.epochs} '
               f'--gen_every 500 '
               f'--gen_sentence "{gen_text}"')
        rc = run_cmd(cmd, f"Step 2: Fine-tuning CSM-1B ({args.epochs} epochs)")
        if rc == 0:
            print(f"\nPhase 3 Complete! Model saved.")
    else:
        print("ERROR: No tokenized data found.")

if __name__ == "__main__":
    main()
