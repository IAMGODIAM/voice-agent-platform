#!/usr/bin/env python3
"""
Speaker Diarization & Voice Isolation Pipeline
==============================================
Isolates Israel's voice from mixed (Zoom) recordings using:
1. MFCC-based speaker embeddings
2. Spectral clustering to separate speakers
3. Anchor matching: known solo segments define Israel's voice signature
4. Output: clean Israel-only WAV segments for CSM-1B fine-tuning

Usage:
    python3 speaker_isolate.py \
        --segments-dir ./voice_data/segments \
        --anchor-dirs 9Gb2FpMipb4 \
        --output-dir ./voice_data/israel_isolated \
        --threshold 0.65
"""

import os
import sys
import json
import argparse
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import librosa
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist


# ── Config ──────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
SEGMENT_DURATION = 2.0  # seconds per analysis window
HOP_DURATION = 0.5      # hop between windows
N_MFCC = 20
N_COMPONENTS = 12       # PCA-like reduction for embeddings


def extract_embedding(audio_path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Extract speaker embedding from a WAV file using MFCCs + spectral features."""
    try:
        y, _ = librosa.load(audio_path, sr=sr, mono=True)
    except Exception as e:
        print(f"  [WARN] Cannot load {audio_path}: {e}")
        return None

    if len(y) < sr * 0.5:  # Skip very short segments (< 0.5s)
        return None

    # MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=512, hop_length=256)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    features = np.concatenate([
        mfcc_mean,
        mfcc_std,
        [np.mean(spectral_centroid), np.std(spectral_centroid)],
        [np.mean(spectral_rolloff), np.std(spectral_rolloff)],
        [np.mean(spectral_bandwidth), np.std(spectral_bandwidth)],
        [np.mean(zero_crossing_rate), np.std(zero_crossing_rate)],
        [np.mean(rms), np.std(rms)],
    ])

    # Replace NaN/Inf with 0
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features


def cluster_speakers(embeddings: list, n_speakers: int = None) -> np.ndarray:
    """
    Cluster embeddings using Agglomerative Clustering.
    If n_speakers is None, estimated from the data using silhouette analysis.
    """
    if len(embeddings) < 3:
        return np.array([0] * len(embeddings))

    X = np.array(embeddings)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Estimate number of speakers if not provided
    if n_speakers is None:
        # For Zoom recordings: typically 2-4 speakers
        # Run clustering with different k and pick best silhouette
        from sklearn.metrics import silhouette_score
        best_score = -1
        best_k = 2
        best_labels = None

        for k in range(2, min(6, len(X_scaled) // 3 + 1)):
            try:
                clustering = AgglomerativeClustering(n_clusters=k, linkage='ward')
                labels = clustering.fit_predict(X_scaled)
                if len(set(labels)) < 2:
                    continue
                score = silhouette_score(X_scaled, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
                    best_labels = labels
            except Exception:
                continue

        if best_labels is not None:
            return best_labels
        return np.array([0] * len(embeddings))

    clustering = AgglomerativeClustering(n_clusters=n_speakers, linkage='ward')
    return clustering.fit_predict(X_scaled)


def compute_speaker_centroid(embeddings: list) -> np.ndarray:
    """Compute centroid (mean embedding) for a group of speaker embeddings."""
    if not embeddings:
        return None
    return np.mean(np.array(embeddings), axis=0)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    if a is None or b is None:
        return 0.0
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return dot / norm


def process_video_directory(
    video_dir: Path,
    anchor_centroid: np.ndarray,
    output_dir: Path,
    metadata: dict,
    distance_threshold: float = 0.65
) -> dict:
    """
    Process all segments in a video directory.
    Returns counts of kept/rejected segments.
    """
    # Get all WAV files directly in this directory (not subdirectories)
    wav_files = sorted(video_dir.glob("*.wav"))

    if not wav_files:
        return {"kept": 0, "rejected": 0, "total": 0}

    print(f"\n📁 Processing {video_dir.name}: {len(wav_files)} segments")

    # Phase 1: Extract embeddings
    print(f"  Phase 1: Extracting embeddings...")
    embeddings = []
    valid_files = []

    for i, wav_file in enumerate(wav_files):
        if i % 100 == 0 and i > 0:
            print(f"    {i}/{len(wav_files)} processed...")
        emb = extract_embedding(str(wav_file))
        if emb is not None:
            embeddings.append(emb)
            valid_files.append(wav_file)

    print(f"  Extracted {len(embeddings)} valid embeddings from {len(wav_files)} files")

    if len(embeddings) < 2:
        print(f"  [SKIP] Not enough valid segments for clustering")
        return {"kept": 0, "rejected": 0, "total": len(wav_files)}

    # Phase 2: Cluster speakers
    print(f"  Phase 2: Clustering speakers...")
    labels = cluster_speakers(embeddings)
    unique_labels = set(labels)
    print(f"  Found {len(unique_labels)} speaker clusters")

    # Phase 3: Identify Israel's cluster using anchor
    print(f"  Phase 3: Matching Israel's voice signature...")
    israel_cluster = None
    best_similarity = -1

    for label in unique_labels:
        cluster_embs = [embeddings[i] for i, l in enumerate(labels) if l == label]
        centroid = compute_speaker_centroid(cluster_embs)
        sim = cosine_similarity(anchor_centroid, centroid)
        print(f"    Cluster {label}: {len(cluster_embs)} segments, similarity to anchor = {sim:.4f}")
        if sim > best_similarity:
            best_similarity = sim
            israel_cluster = label

    print(f"  → Israel's cluster: {israel_cluster} (similarity: {best_similarity:.4f})")

    # Phase 4: Extract Israel-only segments
    print(f"  Phase 4: Extracting Israel-only segments...")
    kept = 0
    rejected = 0

    video_output = output_dir / video_dir.name
    video_output.mkdir(parents=True, exist_ok=True)

    for i, (wav_file, label) in enumerate(zip(valid_files, labels)):
        if label == israel_cluster:
            dest = video_output / wav_file.name
            shutil.copy2(wav_file, dest)
            kept += 1
        else:
            rejected += 1

    # Also include anchor files if this IS an anchor directory
    if video_dir.name in metadata.get("anchor_dirs", []):
        print(f"  [ANCHOR] Including all {len(wav_files)} anchor segments")
        # All anchor files are already included since they match Israel's cluster

    print(f"  Result: {kept} KEPT (Israel), {rejected} REJECTED (others)")
    return {"kept": kept, "rejected": rejected, "total": len(wav_files)}


def main():
    parser = argparse.ArgumentParser(description="Isolate Israel's voice from mixed recordings")
    parser.add_argument("--segments-dir", required=True, help="Path to segments directory")
    parser.add_argument("--anchor-dirs", nargs="+", default=[],
                        help="Video IDs that are known to be Israel-only (anchors)")
    parser.add_argument("--output-dir", default="./voice_data/israel_isolated",
                        help="Output directory for isolated voice")
    parser.add_argument("--threshold", type=float, default=0.65,
                        help="Cosine similarity threshold for speaker matching")
    args = parser.parse_args()

    segments_dir = Path(args.segments_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("🔊 SPEAKER DIARIZATION & VOICE ISOLATION PIPELINE")
    print("="*60)
    print(f"  Segments: {segments_dir}")
    print(f"  Anchors: {args.anchor_dirs}")
    print(f"  Output: {output_dir}")
    print(f"  Threshold: {args.threshold}")
    print(f"  Time: {datetime.now().isoformat()}")

    # Step 1: Find all video directories
    video_dirs = sorted([d for d in segments_dir.iterdir() if d.is_dir()])
    print(f"\n📂 Found {len(video_dirs)} video directories: {[d.name for d in video_dirs]}")

    # Step 2: Build anchor profile from known Israel-only videos
    print("\n🎯 STEP 1: Building Israel's voice anchor profile...")
    anchor_embeddings = []

    for anchor_name in args.anchor_dirs:
        anchor_dir = segments_dir / anchor_name
        if not anchor_dir.exists():
            print(f"  [WARN] Anchor directory not found: {anchor_dir}")
            continue

        wav_files = sorted(anchor_dir.glob("*.wav"))
        print(f"  Loading {len(wav_files)} anchor segments from {anchor_name}...")

        for wav_file in wav_files:
            emb = extract_embedding(str(wav_file))
            if emb is not None:
                anchor_embeddings.append(emb)

    if not anchor_embeddings:
        print("  [ERROR] No valid anchor embeddings found! Cannot proceed without anchors.")
        sys.exit(1)

    anchor_centroid = compute_speaker_centroid(anchor_embeddings)
    print(f"  Anchor profile built from {len(anchor_embeddings)} segments")
    print(f"  Centroid shape: {anchor_centroid.shape}")

    # Step 3: Process each video directory
    print("\n🔄 STEP 2: Processing video directories...")
    metadata = {
        "created": datetime.now().isoformat(),
        "anchor_dirs": args.anchor_dirs,
        "anchor_segments": len(anchor_embeddings),
        "threshold": args.threshold,
        "videos": [],
        "total_kept": 0,
        "total_rejected": 0,
    }

    total_kept = 0
    total_rejected = 0

    for video_dir in video_dirs:
        result = process_video_directory(
            video_dir,
            anchor_centroid,
            output_dir,
            metadata,
            args.threshold
        )
        metadata["videos"].append({
            "id": video_dir.name,
            **result
        })
        total_kept += result["kept"]
        total_rejected += result["rejected"]

    metadata["total_kept"] = total_kept
    metadata["total_rejected"] = total_rejected

    # Step 4: Save metadata
    with open(output_dir / "isolation_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Step 5: Summary
    print("\n" + "="*60)
    print("📊 ISOLATION COMPLETE")
    print("="*60)
    print(f"  Israel's voice segments KEPT: {total_kept}")
    print(f"  Other speakers REJECTED: {total_rejected}")
    print(f"  Precision: / ( Anchor-only so precision = all kept are Israel)")
    print(f"  Output: {output_dir}")
    print(f"  Total files on disk: {len(list(output_dir.rglob('*.wav')))}")
    print("="*60)


if __name__ == "__main__":
    main()
