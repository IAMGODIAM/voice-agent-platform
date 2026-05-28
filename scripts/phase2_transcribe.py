#!/usr/bin/env python3
"""
Phase 2: Transcribe clean Israel audio with Whisper
"""
import os
import json
import whisper

WORKSPACE = "C:/Users/User/Desktop/israel-voice-workspace"
CLEAN_DIR = os.path.join(WORKSPACE, "voice_data", "israel", "clean_israel")
TRANS_DIR = os.path.join(WORKSPACE, "voice_data", "israel", "transcripts")
META_DIR = os.path.join(WORKSPACE, "voice_data", "israel", "metadata")

os.makedirs(TRANS_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)

# Find all clean Israel WAV files
clean_files = [f for f in os.listdir(CLEAN_DIR) if f.endswith("_israel.wav")]
print(f"Found {len(clean_files)} clean Israel audio files")

# Load Whisper
print("Loading Whisper large-v3...")
model = whisper.load_model("large-v3")
print("Whisper loaded!")

all_metadata = []

for clean_file in clean_files:
    video_id = clean_file.replace("_israel.wav", "")
    clean_path = os.path.join(CLEAN_DIR, clean_file)
    trans_path = os.path.join(TRANS_DIR, f"{video_id}_transcript.json")

    print(f"\nTranscribing: {video_id}")
    result = model.transcribe(clean_path, language="en", verbose=False)

    entries = []
    for seg in result.get("segments", []):
        text = seg.get("text", "").strip()
        if len(text) < 5:
            continue
        entry = {
            "start": seg.get("start", 0),
            "end": seg.get("end", 0),
            "text": text,
            "path": clean_path,
            "speaker": 0,
        }
        entries.append(entry)
        all_metadata.append(entry)

    with open(trans_path, "w") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    print(f"  {len(entries)} segments, {sum(len(e['text']) for e in entries)} chars")

# Split and save metadata for sesame-finetune
split_idx = int(len(all_metadata) * 0.9)
train_meta = all_metadata[:split_idx]
val_meta = all_metadata[split_idx:]

with open(os.path.join(META_DIR, "train.json"), "w") as f:
    json.dump(train_meta, f, indent=2, ensure_ascii=False)
with open(os.path.join(META_DIR, "val.json"), "w") as f:
    json.dump(val_meta, f, indent=2, ensure_ascii=False)

print(f"\n{'='*60}")
print(f"Phase 2 Complete!")
print(f"  Train: {len(train_meta)} samples")
print(f"  Val: {len(val_meta)} samples")
print(f"  Metadata saved to: {META_DIR}")
print(f"{'='*60}")
