#!/bin/bash
# ============================================================
# Israel Voice Pipeline — End-to-End Runner
# Execute on MC (Monte-Cristo) with RTX 3070
# ============================================================
set -euo pipefail

echo "═══════════════════════════════════════════════════"
echo "  ISRAEL VOICE PIPELINE — CSM-1B Fine-Tuning"
echo "═══════════════════════════════════════════════════"

# ── Configuration ──────────────────────────────────────────
VOICE_WORKSPACE="/home/user/hermes-workspace/voice-agent-platform"
VOICE_DATA="${VOICE_WORKSPACE}/voice_data/israel"
REFERENCE_AUDIO="${VOICE_DATA}/reference_israel.wav"  # Chairman's reference sample
SEGMENTS_DIR="${VOICE_DATA}/segments"
DIARIZE_OUTPUT="${VOICE_DATA}/diarization"
CLEAN_OUTPUT="${VOICE_DATA}/clean_israel"
TOKENIZED_OUTPUT="${VOICE_DATA}/tokens.hdf5"
TRAIN_META="${VOICE_DATA}/train_metadata.json"
VAL_META="${VOICE_DATA}/val_metadata.json"
FINETUNE_OUTPUT="${VOICE_WORKSPACE}/voice-agent-platform/output/israel_csm"

# ── Phase 1: Speaker Diarization ──────────────────────────
echo ""
echo "━━━ Phase 1: Speaker Diarization ━━━"

pip install pyannote.audio 2>/dev/null || true

python3 << 'PYEOF'
import os, json, torch, glob
from pyannote.audio import Pipeline
from pathlib import Path

SEGMENTS_DIR = os.environ.get("SEGMENTS_DIR", "/home/user/hermes-workspace/voice-agent-platform/voice_data/israel/segments")
DIARIZE_OUTPUT = os.environ.get("DIARIZE_OUTPUT", "/home/user/hermes-workspace/voice-agent-platform/voice_data/israel/diarization")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

print(f"[Diarization] Loading pyannote pipeline...")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HF_TOKEN if HF_TOKEN else None
)
pipeline.to(torch.device("cuda"))
print(f"[Diarization] Pipeline loaded on GPU")

# Find all audio files in segments directory
audio_files = glob.glob(f"{SEGMENTS_DIR}/**/*.wav", recursive=True)
# Also check for original video audio files
video_audio = glob.glob(f"{SEGMENTS_DIR}/../audio/*.wav")

# Group by video (parent dir)
from collections import defaultdict
video_files = defaultdict(list)
for f in audio_files:
    video_id = Path(f).parent.name
    video_files[video_id].append(f)

# For diarization, we need the FULL video audio, not segments
# Look for full audio files
full_audio_files = glob.glob(f"{SEGMENTS_DIR}/../audio/*.wav") + glob.glob(f"{SEGMENTS_DIR}/../audio/*.mp3") + glob.glob(f"{SEGMENTS_DIR}/../audio/*.m4a")

if not full_audio_files:
    print("[Diarization] WARNING: No full video audio found. Will diarize individual segments instead.")
    # Fallback: diarize each segment batch
    batch_size = 50
    all_diarizations = {}
    for video_id, files in video_files.items():
        print(f"[Diarization] Processing video: {video_id} ({len(files)} segments)")
        for i in range(0, len(files), batch_size):
            batch = files[i:i+batch_size]
            for audio_file in batch:
                try:
                    diarization = pipeline(audio_file)
                    segments = []
                    for turn, _, speaker in diarization.itertracks(yield_label=True):
                        segments.append({
                            "start": turn.start,
                            "end": turn.end,
                            "speaker": speaker
                        })
                    all_diarizations[audio_file] = segments
                except Exception as e:
                    print(f"[Diarization] Error on {audio_file}: {e}")
else:
    print(f"[Diarization] Found {len(full_audio_files)} full video audio files")
    all_diarizations = {}
    for audio_file in full_audio_files:
        print(f"[Diarization] Processing: {audio_file}")
        try:
            diarization = pipeline(audio_file)
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker
                })
            all_diarizations[audio_file] = segments
            print(f"  → {len(segments)} speaker turns detected")
        except Exception as e:
            print(f"[Diarization] Error on {audio_file}: {e}")

# Save diarization results
os.makedirs(DIARIZE_OUTPUT, exist_ok=True)
with open(f"{DIARIZE_OUTPUT}/diarization_results.json", "w") as f:
    json.dump(all_diarizations, f, indent=2)

print(f"[Diarization] Complete. Results saved to {DIARIZE_OUTPUT}/diarization_results.json")
PYEOF

echo "✅ Phase 1 complete"

# ── Phase 2: Voice Isolation ───────────────────────────────
echo ""
echo "━━━ Phase 2: Voice Isolation (Reference Matching) ━━━"

python3 << 'PYEOF'
import os, json, torch, glob, numpy as np
import torchaudio
from pyannote.audio import Model as EmbeddingModel
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

REFERENCE_AUDIO = os.environ.get("REFERENCE_AUDIO", "")
DIARIZE_OUTPUT = os.environ.get("DIARIZE_OUTPUT", "/home/user/hermes-workspace/voice-agent-platform/voice_data/israel/diarization")
CLEAN_OUTPUT = os.environ.get("CLEAN_OUTPUT", "/home/user/hermes-workspace/voice-agent-platform/voice_data/israel/clean_israel")
SEGMENTS_DIR = os.environ.get("SEGMENTS_DIR", "/home/user/hermes-workspace/voice-agent-platform/voice_data/israel/segments")

if not REFERENCE_AUDIO or not os.path.exists(REFERENCE_AUDIO):
    print("[VoiceIsolation] WARNING: Reference audio not found at", REFERENCE_AUDIO)
    print("[VoiceIsolation] Skipping reference matching — will use dominant speaker heuristic")
    # Find dominant speaker from diarization
    diarize_file = f"{DIARIZE_OUTPUT}/diarization_results.json"
    if os.path.exists(diarize_file):
        with open(diarize_file) as f:
            diarizations = json.load(f)
        speaker_durations = {}
        for audio_file, segments in diarizations.items():
            for seg in segments:
                spk = seg["speaker"]
                dur = seg["end"] - seg["start"]
                speaker_durations[spk] = speaker_durations.get(spk, 0) + dur
        israel_speaker = max(speaker_durations, key=speaker_durations.get)
        print(f"[VoiceIsolation] Dominant speaker: {israel_speaker} ({speaker_durations[israel_speaker]:.1f}s)")
    else:
        israel_speaker = "SPEAKER_00"
        print("[VoiceIsolation] No diarization results, defaulting to SPEAKER_00")
else:
    print(f"[VoiceIsolation] Loading embedding model...")
    embedding_model = EmbeddingModel.from_pretrained(
        "pyannote/embedding",
        use_auth_token=os.environ.get("HF_TOKEN", None)
    ).to(torch.device("cuda"))
    
    # Extract reference embedding
    ref_waveform, ref_sr = torchaudio.load(REFERENCE_AUDIO)
    ref_waveform = torchaudio.functional.resample(ref_waveform, ref_sr, 16000)
    ref_embedding = embedding_model(ref_waveform.unsqueeze(0).cuda())
    ref_embedding = ref_embedding.detach().cpu().numpy()
    
    print(f"[VoiceIsolation] Reference embedding extracted: {ref_embedding.shape}")
    
    # Load diarization results
    diarize_file = f"{DIARIZE_OUTPUT}/diarization_results.json"
    with open(diarize_file) as f:
        diarizations = json.load(f)
    
    # Compare reference against each speaker's segments
    speaker_similarities = {}
    for audio_file, segments in diarizations.items():
        for seg in segments:
            spk = seg["speaker"]
            if spk not in speaker_similarities:
                speaker_similarities[spk] = {"total_sim": 0, "count": 0}
            # Extract embedding for this segment
            try:
                wav, sr = torchaudio.load(audio_file, 
                    frame_offset=int(seg["start"] * sr),
                    num_frames=int((seg["end"] - seg["start"]) * sr))
                wav = torchaudio.functional.resample(wav, sr, 16000)
                seg_emb = embedding_model(wav.unsqueeze(0).cuda())
                seg_emb = seg_emb.detach().cpu().numpy()
                sim = cosine_similarity(ref_embedding, seg_emb)[0][0]
                speaker_similarities[spk]["total_sim"] += sim
                speaker_similarities[spk]["count"] += 1
            except:
                pass
    
    # Pick speaker with highest average similarity
    best_speaker = None
    best_sim = -1
    for spk, stats in speaker_similarities.items():
        avg_sim = stats["total_sim"] / max(stats["count"], 1)
        print(f"  Speaker {spk}: avg similarity = {avg_sim:.4f} (from {stats['count']} segments)")
        if avg_sim > best_sim:
            best_sim = avg_sim
            best_speaker = spk
    
    israel_speaker = best_speaker
    print(f"\n[VoiceIsolation] ✅ Israel identified as: {israel_speaker} (similarity: {best_sim:.4f})")

# Export: save israel_speaker ID for next phase
os.makedirs(CLEAN_OUTPUT, exist_ok=True)
with open(f"{CLEAN_OUTPUT}/israel_speaker_id.txt", "w") as f:
    f.write(israel_speaker)

print(f"[VoiceIsolation] Speaker ID saved: {israel_speaker}")
PYEOF

echo "✅ Phase 2 complete"

# ── Phase 3: Transcription (WhisperX) ─────────────────────
echo ""
echo "━━━ Phase 3: Transcription ━━━"

pip install whisperx 2>/dev/null || true

python3 << 'PYEOF'
import os, json, glob
from pathlib import Path

CLEAN_OUTPUT = os.environ.get("CLEAN_OUTPUT", "/home/user/hermes-workspace/voice-agent-platform/voice_data/israel/clean_israel")
DIARIZE_OUTPUT = os.environ.get("DIARIZE_OUTPUT", "/home/user/hermes-workspace/voice-agent-platform/voice_data/israel/diarization")
SEGMENTS_DIR = os.environ.get("SEGMENTS_DIR", "/home/user/hermes-workspace/voice-agent-platform/voice_data/israel/segments")
TRAIN_META = os.environ.get("TRAIN_META", "/home/user/hermes-workspace/voice-agent-platform/voice_data/israel/train_metadata.json")
VAL_META = os.environ.get("VAL_META", "/home/user/hermes-workspace/voice-agent-platform/voice_data/israel/val_metadata.json")

# Read Israel's speaker ID
with open(f"{CLEAN_OUTPUT}/israel_speaker_id.txt") as f:
    israel_speaker = f.read().strip()

print(f"[Transcription] Israel speaker ID: {israel_speaker}")

# Load diarization results
diarize_file = f"{DIARIZE_OUTPUT}/diarization_results.json"
with open(diarize_file) as f:
    diarizations = json.load(f)

# Extract Israel's segments
israel_segments = []
for audio_file, segments in diarizations.items():
    for seg in segments:
        if seg["speaker"] == israel_speaker:
            dur = seg["end"] - seg["start"]
            if 0.5 <= dur <= 30.0:  # Quality filter
                israel_segments.append({
                    "audio_file": audio_file,
                    "start": seg["start"],
                    "end": seg["end"],
                    "duration": dur
                })

print(f"[Transcription] Found {len(israel_segments)} clean segments for Israel")

# Run WhisperX transcription
try:
    import whisperx
    device = "cuda"
    model = whisperx.load_model("large-v3", device, compute_type="float16")
    
    metadata = []
    for i, seg in enumerate(israel_segments):
        print(f"[Transcription] Processing segment {i+1}/{len(israel_segments)}...")
        
        # Load audio segment
        import torchaudio
        wav, sr = torchaudio.load(
            seg["audio_file"],
            frame_offset=int(seg["start"] * sr),
            num_frames=int(seg["duration"] * sr)
        )
        wav = torchaudio.functional.resample(wav, sr, 16000)
        
        # Save temp wav
        temp_wav = f"{CLEAN_OUTPUT}/temp_seg_{i:04d}.wav"
        torchaudio.save(temp_wav, wav, 16000)
        
        # Transcribe
        result = model.translate(temp_wav) if hasattr(model, 'translate') else model.transcribe(temp_wav)
        text = result["text"].strip() if isinstance(result, dict) else str(result)
        
        # Clean up temp file
        os.remove(temp_wav)
        
        if len(text) > 5:  # Skip very short/empty transcriptions
            metadata.append({
                "text": text,
                "path": temp_wav.replace("_temp_", "_"),
                "speaker": 999,
                "start": seg["start"],
                "end": seg["end"]
            })
    
    print(f"[Transcription] Transcribed {len(metadata)} segments")
    
except ImportError:
    print("[Transcription] WhisperX not available. Using placeholder transcriptions.")
    print("[Transcription] Install whisperx: pip install whisperx")
    # Fallback: create metadata without transcriptions
    metadata = []
    for seg in israel_segments:
        metadata.append({
            "text": "[TRANSCRIPTION NEEDED]",
            "path": seg["audio_file"],
            "speaker": 999,
            "start": seg["start"],
            "end": seg["end"]
        })

# 80/20 split
import random
random.seed(42)
random.shuffle(metadata)
split_idx = int(len(metadata) * 0.8)
train_data = metadata[:split_idx]
val_data = metadata[split_idx:]

with open(TRAIN_META, "w") as f:
    json.dump(train_data, f, indent=2)
with open(VAL_META, "w") as f:
    json.dump(val_data, f, indent=2)

print(f"[Transcription] Train: {len(train_data)} samples, Val: {len(val_data)} samples")
print(f"[Transcription] Metadata saved to {TRAIN_META} and {VAL_META}")
PYEOF

echo "✅ Phase 3 complete"

# ── Phase 4: Pre-tokenization ──────────────────────────────
echo ""
echo "━━━ Phase 4: Pre-tokenization ━━━"

cd "${VOICE_WORKSPACE}/voice-agent-platform/sesame-finetune"
python3 pretokenize.py \
  --train_data "${TRAIN_META}" \
  --val_data "${VAL_META}" \
  --output "${TOKENIZED_OUTPUT}" \
  --device cuda \
  --save_every 100

echo "✅ Phase 4 complete"

# ── Phase 5: Fine-Tuning ───────────────────────────────────
echo ""
echo "━━━ Phase 5: CSM-1B Fine-Tuning ━━━"

python3 train.py \
  --data "${TOKENIZED_OUTPUT}" \
  --config ./configs/israel_finetune.yaml \
  --model_name_or_checkpoint_path "sesame/csm-1b" \
  --n_epochs 25 \
  --gen_every 500 \
  --gen_sentence "God Bless the Child that's got its own." \
  --gen_speaker 999 \
  --use_amp \
  --wandb_project "csm-israel-voice" \
  --wandb_name "israel-v1" \
  --output_dir "${FINETUNE_OUTPUT}"

echo "✅ Phase 5 complete"

# ── Phase 6: Evaluation ────────────────────────────────────
echo ""
echo "━━━ Phase 6: Evaluation ━━━"
echo "Check W&B dashboard for loss curves and generated audio samples"
echo "Best model: ${FINETUNE_OUTPUT}/model_bestval.pt"

echo ""
echo "═══════════════════════════════════════════════════"
echo "  PIPELINE COMPLETE"
echo "═══════════════════════════════════════════════════"
