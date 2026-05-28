#!/usr/bin/env python3
"""
Speaker Diarization & Voice Isolation Pipeline v2
==================================================
Isolates Israel's voice from mixed (Zoom) recordings.

Strategy:
- Solo videos (Campaign to Movement) = anchor profile for Israel's voice
- Podcast videos (Justice is Here) = mixed, need diarization
- Uses MFCC + spectral features + Agglomerative Clustering
- Cosine similarity matching against Israel's anchor centroid

File structure: all WAVs in one directory, named {video_id}_{index}_{seg}.wav
"""

import os
import sys
import json
import argparse
import shutil
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
import librosa
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cosine as cosine_distance


# ── Config ──────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
N_MFCC = 20
MIN_SEGMENT_DURATION = 0.5  # seconds — skip shorter


def extract_embedding(audio_path: str):
    """Extract speaker embedding from a WAV file. Returns 50-dim feature vector or None."""
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    except Exception:
        return None, None

    if len(y) < int(SAMPLE_RATE * MIN_SEGMENT_DURATION):
        return None, None

    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    mfcc_skew = np.zeros(N_MFCC)
    for i in range(N_MFCC):
        vals = mfcc[i]
        if np.std(vals) > 1e-7:
            mfcc_skew[i] = float(np.mean(((vals - np.mean(vals)) / np.std(vals)) ** 3))

    # Spectral
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)
    sr9 = librosa.feature.spectral_rolloff(y=y, sr=sr)
    sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    # Pitch (F0) using piptrack — distinctive for speaker ID
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=60, fmax=400)
    pitch_vals = []
    for t in range(pitches.shape[1]):
        idx = np.argmax(magnitudes[:, t])
        if magnitudes[idx, t] > 0:
            pitch_vals.append(pitches[idx, t])
    pitch_mean = np.mean(pitch_vals) if pitch_vals else 0.0
    pitch_std = np.std(pitch_vals) if pitch_vals else 0.0

    features = np.concatenate([
        mfcc_mean,          # 20
        mfcc_std,           # 20
        [np.mean(sc), np.std(sc)],
        [np.mean(sr9), np.std(sr9)],
        [np.mean(sb), np.std(sb)],
        [np.mean(zcr), np.std(zcr)],
        [np.mean(rms), np.std(rms)],
        [pitch_mean, pitch_std],
        mfcc_skew,          # 20 — total = 68
    ])

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # L2-normalize
    norm = np.linalg.norm(features)
    if norm > 0:
        features = features / norm

    return features, float(len(y) / sr)


def build_anchor_profile(segment_dir, anchor_video_ids):
    """Build Israel's voice centroid from known solo recordings."""
    embeddings = []
    durations = []

    for f in sorted(Path(segment_dir).glob("*.wav")):
        # Check if this file belongs to an anchor video
        is_anchor = any(f.name.startswith(vid) for vid in anchor_video_ids)
        if not is_anchor:
            continue

        emb, dur = extract_embedding(str(f))
        if emb is not None:
            embeddings.append(emb)
            durations.append(dur)

    if not embeddings:
        return None, 0

    return np.mean(np.array(embeddings), axis=0), len(embeddings)


def diarize_and_isolate(segment_dir, mixed_video_ids, anchor_centroid, output_dir, threshold=0.6):
    """
    For each mixed video:
    1. Extract embeddings from all segments
    2. Cluster into speakers (2-4)
    3. Match Israel's cluster via cosine similarity to anchor
    4. Copy Israel-only segments to output
    """
    results = {}

    for vid in mixed_video_ids:
        print(f"\n🎙️ Diarizing: {vid}")
        wav_files = sorted(Path(segment_dir).glob(f"{vid}_*.wav"))
        print(f"  Found {len(wav_files)} segments")

        if len(wav_files) < 5:
            print(f"  [SKIP] Too few segments")
            results[vid] = {"kept": 0, "rejected": 0, "total": len(wav_files)}
            continue

        # Extract embeddings
        embeddings = []
        valid_files = []
        for i, wf in enumerate(wav_files):
            emb, dur = extract_embedding(str(wf))
            if emb is not None:
                embeddings.append(emb)
                valid_files.append(wf)

        n_valid = len(embeddings)
        print(f"  Valid embeddings: {n_valid}/{len(wav_files)}")

        if n_valid < 5:
            results[vid] = {"kept": 0, "rejected": len(wav_files), "total": len(wav_files)}
            continue

        # Cluster speakers — try k=2,3,4 and pick best silhouette
        X = np.array(embeddings)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        from sklearn.metrics import silhouette_score
        best_k = 2
        best_score = -1
        best_labels = np.array([0] * n_valid)

        max_k = min(5, n_valid // 3)
        for k in range(2, max_k + 1):
            try:
                clustering = AgglomerativeClustering(n_clusters=k, linkage='average')
                labels = clustering.fit_predict(X_scaled)
                uniq = set(labels)
                if len(uniq) < 2:
                    continue
                score = silhouette_score(X_scaled, labels)
                print(f"    k={k}: silhouette={score:.4f}")
                if score > best_score:
                    best_score = score
                    best_k = k
                    best_labels = labels
            except Exception as e:
                print(f"    k={k}: failed ({e})")
                continue

        print(f"  Best: k={best_k} (silhouette={best_score:.4f})")

        # Match Israel's cluster
        israel_cluster = -1
        best_sim = -1
        unique_labels = sorted(set(best_labels))

        for label in unique_labels:
            mask = best_labels == label
            cluster_embs = X[mask]
            centroid = np.mean(cluster_embs, axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            sim = 1.0 - cosine_distance(anchor_centroid, centroid)
            n_segs = int(np.sum(mask))
            print(f"    Cluster {label}: {n_segs} segs, sim={sim:.4f}")
            if sim > best_sim:
                best_sim = sim
                israel_cluster = label

        print(f"  → Israel = cluster {israel_cluster} (sim={best_sim:.4f})")

        # Extract
        vid_output = Path(output_dir) / vid
        vid_output.mkdir(parents=True, exist_ok=True)

        kept = 0
        rejected = 0
        for wf, label in zip(valid_files, best_labels):
            if label == israel_cluster and best_sim >= threshold:
                shutil.copy2(wf, vid_output / wf.name)
                kept += 1
            else:
                rejected += 1

        # Rejected due to invalid embeddings
        rejected += len(wav_files) - n_valid

        print(f"  ✅ KEPT: {kept} | ❌ REJECTED: {rejected}")
        results[vid] = {"kept": kept, "rejected": rejected, "total": len(wav_files)}

    return results


def main():
    parser = argparse.ArgumentParser(description="Isolate Israel's voice from mixed recordings")
    parser.add_argument("--segments-dir", required=True)
    parser.add_argument("--anchor-videos", nargs="+", required=True,
                        help="Video IDs known to be Israel-only")
    parser.add_argument("--mixed-videos", nargs="+", required=True,
                        help="Video IDs that are mixed/Zoom recordings")
    parser.add_argument("--output-dir", default="./voice_data/israel_isolated")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="Min cosine similarity to classify as Israel")
    args = parser.parse_args()

    segment_dir = Path(args.segments_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🔊 SPEAKER DIARIZATION & VOICE ISOLATION")
    print("=" * 60)
    print(f"  Time: {datetime.now().isoformat()}")
    print(f"  Segments dir: {segment_dir}")
    print(f"  Anchors: {args.anchor_videos}")
    print(f"  Mixed: {args.mixed_videos}")
    print(f"  Threshold: {args.threshold}")

    # Build anchor
    print("\n🎯 Building Israel's voice anchor profile...")
    anchor_centroid, n_anchor = build_anchor_profile(segment_dir, args.anchor_videos)
    if anchor_centroid is None:
        print("[FATAL] No anchor segments found!")
        sys.exit(1)
    print(f"  Anchor: {n_anchor} segments, centroid norm={np.linalg.norm(anchor_centroid):.4f}")

    # Diarize mixed videos
    print("\n🔄 Diarizing mixed recordings...")
    results = diarize_and_isolate(
        segment_dir, args.mixed_videos, anchor_centroid, output_dir, args.threshold
    )

    # Summary
    total_kept = sum(r["kept"] for r in results.values())
    total_rejected = sum(r["rejected"] for r in results.values())
    total_anchor = sum(
        len(list(segment_dir.glob(f"{vid}_*.wav"))) for vid in args.anchor_videos
    )

    print("\n" + "=" * 60)
    print("📊 FINAL RESULTS")
    print("=" * 60)
    print(f"  Anchor segments (Israel solo): {total_anchor}")
    print(f"  Isolated from mixed (Israel):  {total_kept}")
    print(f"  Rejected (other speakers):     {total_rejected}")
    print(f"  Total Israel voice segments:   {total_anchor + total_kept}")
    print(f"  Output: {output_dir}")
    print(f"  Files on disk: {len(list(output_dir.rglob('*.wav')))}")
    print("=" * 60)

    # Save metadata
    metadata = {
        "created": datetime.now().isoformat(),
        "anchor_videos": args.anchor_videos,
        "mixed_videos": args.mixed_videos,
        "n_anchor_segments": n_anchor,
        "results": results,
        "total_israel_segments": total_anchor + total_kept,
        "threshold": args.threshold,
    }
    with open(output_dir / "isolation_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n📝 Metadata saved to {output_dir / 'isolation_metadata.json'}")


if __name__ == "__main__":
    main()
