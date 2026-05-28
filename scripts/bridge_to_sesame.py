#!/usr/bin/env python3
"""
Bridge: Diarization + Transcription -> metadata.json for sesame-finetune
=======================================================================
Takes the 4 source WAVs, runs MFCC diarization to isolate Israel's voice,
transcribes with Whisper, and outputs train.json + val.json in the format
expected by knottwill/sesame-finetune's pretokenize.py.

Usage:
    python bridge_to_sesame.py --workspace C:\Users\User\Desktop\israel-voice-workspace
"""

import os
import sys
import json
import argparse
import numpy as np
import librosa
import soundfile as sf
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

WORKSPACE = "C:/Users/User/Desktop/israel-voice-workspace"
RAW_DIR = os.path.join(WORKSPACE, "voice_data", "raw_audio")
CLEAN_DIR = os.path.join(WORKSPACE, "voice_data", "israel", "clean_israel")
DIAR_DIR = os.path.join(WORKSPACE, "voice_data", "israel", "diarization")
META_DIR = os.path.join(WORKSPACE, "voice_data", "israel", "metadata")
TRANS_DIR = os.path.join(WORKSPACE, "voice_data", "israel", "transcripts")

VIDEO_FILES = ["tymBGoS8s7Q.wav", "9Gb2FpMipb4.wav", "d_qcnkJMGCs.wav", "Aa1zczMPl4.wav"]
SAMPLE_RATE = 16000
SEGMENT_DURATION = 30  # seconds
N_MFCC = 20

def ensure_dirs():
    for d in [CLEAN_DIR, DIAR_DIR, META_DIR, TRANS_DIR]:
        os.makedirs(d, exist_ok=True)

def diarize_audio(wav_path, video_id):
    """MFCC + clustering diarization. Returns Israel's audio and segments."""
    print(f"\n  Diarizing: {video_id}")

    y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
    duration = len(y) / sr
    print(f"    Duration: {duration:.1f}s")

    # Create analysis windows
    seg_len = SEGMENT_DURATION * sr
    segments = []
    for start in range(0, len(y), seg_len):
        end = min(start + seg_len, len(y))
        if end - start < sr * 2:
            continue
        segments.append((start, end))

    print(f"    {len(segments)} analysis windows")

    # Extract MFCC features
    features = []
    for start, end in segments:
        seg_audio = y[start:end]
        mfcc = librosa.feature.mfcc(y=seg_audio, sr=sr, n_mfcc=N_MFCC)
        feat_vector = np.concatenate([
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
            np.median(mfcc, axis=1),
        ])
        features.append(feat_vector)

    features = np.array(features)

    # Cluster
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    clustering = AgglomerativeClustering(n_clusters=2, linkage="ward")
    labels = clustering.fit_predict(features_scaled)

    # Israel = largest cluster
    cluster_durations = {}
    for i, label in enumerate(labels):
        seg_dur = (segments[i][1] - segments[i][0]) / sr
        cluster_durations[label] = cluster_durations.get(label, 0) + seg_dur

    israel_cluster = max(cluster_durations, key=cluster_durations.get)
    print(f"    Cluster durations: {cluster_durations}")
    print(f"    Israel = cluster {israel_cluster} ({cluster_durations[israel_cluster]:.0f}s)")

    # Extract Israel audio
    israel_audio = []
    israel_segments = []
    for i, label in enumerate(labels):
        if label == israel_cluster:
            start, end = segments[i]
            israel_audio.extend(y[start:end])
            israel_segments.append({
                "start": start / sr,
                "end": end / sr,
            })

    # Save clean audio
    clean_path = os.path.join(CLEAN_DIR, f"{video_id}_israel.wav")
    if israel_audio:
        sf.write(clean_path, np.array(israel_audio), sr)
        print(f"    Saved: {clean_path} ({len(israel_audio)/sr:.1f}s)")

    # Save diarization result
    diar_path = os.path.join(DIAR_DIR, f"{video_id}_diarization.json")
    with open(diar_path, "w") as f:
        json.dump({
            "video_id": video_id,
            "israel_cluster": int(israel_cluster),
            "israel_duration": cluster_durations[israel_cluster],
            "segments": israel_segments,
        }, f, indent=2)

    return clean_path, israel_segments

def transcribe_audio(clean_path, video_id):
    """Transcribe using openai-whisper (already installed on MC)."""
    print(f"\n  Transcribing: {video_id}")

    import whisper

    model = whisper.load_model("large-v3")
    print(f"    Whisper large-v3 loaded")

    result = model.transcribe(clean_path, language="en", verbose=False)

    entries = []
    for seg in result.get("segments", []):
        text = seg.get("text", "").strip()
        if len(text) < 5:
            continue
        entries.append({
            "start": seg.get("start", 0),
            "end": seg.get("end", 0),
            "text": text,
        })

    # Save transcript
    trans_path = os.path.join(TRANS_DIR, f"{video_id}_transcript.json")
    with open(trans_path, "w") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)

    print(f"    {len(entries)} segments transcribed")
    return entries

def create_metadata(clean_path, transcript_entries, video_id, metadata_list):
    """Create metadata entries in sesame-finetune format."""
    for entry in transcript_entries:
        text = entry.get("text", "").strip()
        if len(text) < 10:
            continue
        metadata_list.append({
            "path": clean_path,
            "text": text,
            "start": entry.get("start", 0),
            "end": entry.get("end", 0),
            "speaker": 0,
        })

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=WORKSPACE)
    parser.add_argument("--skip-diarization", action="store_true")
    parser.add_argument("--skip-transcription", action="store_true")
    args = parser.parse_args()

    global WORKSPACE, RAW_DIR, CLEAN_DIR, DIAR_DIR, META_DIR, TRANS_DIR
    WORKSPACE = args.workspace
    RAW_DIR = os.path.join(WORKSPACE, "voice_data", "raw_audio")
    CLEAN_DIR = os.path.join(WORKSPACE, "voice_data", "israel", "clean_israel")
    DIAR_DIR = os.path.join(WORKSPACE, "voice_data", "israel", "diarization")
    META_DIR = os.path.join(WORKSPACE, "voice_data", "israel", "metadata")
    TRANS_DIR = os.path.join(WORKSPACE, "voice_data", "israel", "transcripts")

    ensure_dirs()

    print("="*60)
    print("Bridge: Diarization + Transcription -> sesame-finetune metadata")
    print(f"Workspace: {WORKSPACE}")
    print("="*60)

    all_metadata = []

    for wav_file in VIDEO_FILES:
        video_id = wav_file.replace(".wav", "")
        wav_path = os.path.join(RAW_DIR, wav_file)

        if not os.path.exists(wav_path):
            print(f"  WARNING: {wav_file} not found, skipping")
            continue

        # Phase 1: Diarization
        clean_path = os.path.join(CLEAN_DIR, f"{video_id}_israel.wav")
        if not args.skip_diarization or not os.path.exists(clean_path):
            clean_path, segments = diarize_audio(wav_path, video_id)

        # Phase 2: Transcription
        if not args.skip_transcription:
            transcript_path = os.path.join(TRANS_DIR, f"{video_id}_transcript.json")
            if not os.path.exists(transcript_path):
                entries = transcribe_audio(clean_path, video_id)
            else:
                with open(transcript_path) as f:
                    entries = json.load(f)
                print(f"  Loaded existing transcript: {len(entries)} entries")
        else:
            transcript_path = os.path.join(TRANS_DIR, f"{video_id}_transcript.json")
            if os.path.exists(transcript_path):
                with open(transcript_path) as f:
                    entries = json.load(f)
            else:
                entries = []

        # Build metadata
        if entries:
            create_metadata(clean_path, entries, video_id, all_metadata)

    # Split and save
    split_idx = int(len(all_metadata) * 0.9)
    train_meta = all_metadata[:split_idx]
    val_meta = all_metadata[split_idx:]

    train_path = os.path.join(META_DIR, "train.json")
    val_path = os.path.join(META_DIR, "val.json")

    with open(train_path, "w") as f:
        json.dump(train_meta, f, indent=2, ensure_ascii=False)
    with open(val_path, "w") as f:
        json.dump(val_meta, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Bridge complete!")
    print(f"  Train: {len(train_meta)} samples -> {train_path}")
    print(f"  Val:   {len(val_meta)} samples -> {val_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
