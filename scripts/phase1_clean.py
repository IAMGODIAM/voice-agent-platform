import os, sys, json, argparse
import numpy as np
import librosa
import soundfile as sf
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

WORKSPACE = "C:/Users/User/Desktop/israel-voice-workspace"
RAW_DIR = os.path.join(WORKSPACE, "voice_data", "israel")
CLEAN_DIR = os.path.join(WORKSPACE, "voice_data", "israel", "clean_israel")
DIAR_DIR = os.path.join(WORKSPACE, "voice_data", "israel", "diarization")
META_DIR = os.path.join(WORKSPACE, "voice_data", "israel", "metadata")
TRANS_DIR = os.path.join(WORKSPACE, "voice_data", "israel", "transcripts")

VIDEO_FILES = ["tymBGoS8s7Q.wav", "9Gb2FpMipb4.wav", "d_qcnkJMGCs.wav", "Aa1VzczMPl4.wav"]
SAMPLE_RATE = 16000
SEGMENT_DURATION = 30
N_MFCC = 20

for d in [CLEAN_DIR, DIAR_DIR, META_DIR, TRANS_DIR]:
    os.makedirs(d, exist_ok=True)

print("="*60)
print("Phase 1: MFCC Speaker Diarization")
print("="*60)

total_israel = 0

for wav_file in VIDEO_FILES:
    video_id = wav_file.replace(".wav", "")
    wav_path = os.path.join(RAW_DIR, wav_file)
    if not os.path.exists(wav_path):
        print(f"WARNING: {wav_file} not found, skipping")
        continue
    print(f"\nProcessing: {wav_file}")
    y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
    duration = len(y) / sr
    print(f"  Duration: {duration:.1f}s")
    seg_len = SEGMENT_DURATION * sr
    segments = []
    for start in range(0, len(y), seg_len):
        end = min(start + seg_len, len(y))
        if end - start < sr * 2:
            continue
        segments.append((start, end))
    print(f"  {len(segments)} analysis windows")
    features = []
    for start, end in segments:
        seg_audio = y[start:end]
        mfcc = librosa.feature.mfcc(y=seg_audio, sr=sr, n_mfcc=N_MFCC)
        feat_vector = np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1), np.median(mfcc, axis=1)])
        features.append(feat_vector)
    features = np.array(features)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    clustering = AgglomerativeClustering(n_clusters=2, linkage="ward")
    labels = clustering.fit_predict(features_scaled)
    cluster_durations = {}
    for i, label in enumerate(labels):
        seg_dur = (segments[i][1] - segments[i][0]) / sr
        cluster_durations[label] = cluster_durations.get(label, 0) + seg_dur
    israel_cluster = max(cluster_durations, key=cluster_durations.get)
    print(f"  Cluster durations: {cluster_durations}")
    print(f"  Israel = cluster {israel_cluster} ({cluster_durations[israel_cluster]:.0f}s)")
    israel_audio = []
    israel_segments = []
    for i, label in enumerate(labels):
        if label == israel_cluster:
            start, end = segments[i]
            israel_audio.extend(y[start:end])
            israel_segments.append({"start": start/sr, "end": end/sr})
    clean_path = os.path.join(CLEAN_DIR, f"{video_id}_israel.wav")
    if israel_audio:
        sf.write(clean_path, np.array(israel_audio), sr)
        print(f"  Saved: {clean_path} ({len(israel_audio)/sr:.0f}s)")
    diar_path = os.path.join(DIAR_DIR, f"{video_id}_diarization.json")
    with open(diar_path, "w") as f:
        json.dump({"video_id": video_id, "israel_cluster": int(israel_cluster), "israel_duration": cluster_durations[israel_cluster], "segments": israel_segments}, f, indent=2)
    total_israel += cluster_durations[israel_cluster]

print(f"\n{'='*60}")
print(f"Phase 1 Complete: {total_israel:.0f}s ({total_israel/60:.1f}min) of Israel's voice isolated")
print("="*60)
