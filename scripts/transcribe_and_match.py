#!/usr/bin/env python3
"""
Transcribe + Speech Pattern Matching Pipeline
==============================================
Phase 1: Transcribe all WAV segments using Whisper (tiny model, CPU)
Phase 2: Score each transcript against Israel Voice V3 patterns
Phase 3: Classify segments as "Chairman" or "Other"
Phase 4: Isolate verified Chairman segments

Usage:
    python3 transcribe_and_match.py \
        --segments-dir ./voice_data/israel/segments \
        --voice-profile ./israel_voice_v3.md \
        --output-dir ./voice_data/israel_transcribed \
        --threshold 0.6
"""

import os
import sys
import json
import re
import argparse
import shutil
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import whisper


# ── Voice Profile Patterns (from Israel Voice V3) ──────────────────────
# These are the key linguistic markers that distinguish the Chairman's speech

CHAIRMAN_PHRASES = [
    # Pattern 15 - Directive Formula
    r"incumbent on you",
    r"step-by-step.*meticulous",
    r"meticulous attention to detail",
    r"masterfully execute",
    r"expert masterpiece",
    r"review.*assess.*rate.*refine",

    # Pattern 3 - Moral frame before factual
    r"moral.*(frame|stakes|character)",
    r"character.*context.*moral",
    r"right.*wrong.*just",

    # Pattern 6 - Latin mottos
    r"nil satis nisi optimum",
    r"dum spiro.*spero",
    r"while i breathe.*hope",

    # Pattern 7 - Strategic corroboration
    r"confident that if contacted",
    r"readily attest",
    r"will attest",

    # Pattern 9 - Civilizational pivot
    r"liberty city",
    r"food desert",
    r"systemic",
    r"ground-level expression",
    r"communities like this",

    # Pattern 10 - Already-invested authority
    r"i own",
    r"i bought it",
    r"that is not a metaphor",
    r"\$150",
    r"pine bluff",
    r"1\.2 acres",

    # Pattern 11 - Prayer register
    r"the center holds",
    r"the work is underway",
    r"children of god",
    r"solomon's prayer",

    # Pattern 12 - Self-situating humor
    r"marco rubio",
    r"preach on it",

    # Pattern 13 - Seed metaphor
    r"seed.*planted",
    r"planted.*seed",
    r"harvest",

    # Pattern 14 - Credentials as witness
    r"i am a licensed",
    r"real estate agent",
    r"insurance agent",
    r"underwriter",

    # Pattern 16 - Architect audit
    r"good progress.*drifted",
    r"good --",
    r"not good --",
    r"my recommendation",

    # Pattern 17 - Range declaration
    r"iamgodiam",
    r"in-game name",
    r"ign",

    # Pattern 19 - Rising from Liberty City
    r"rising from liberty city",
    r"liberty city.*miami",
    r"zip code 33150",
    r"climate gentrification",

    # Pattern 20 - 5Es
    r"environmental.*economic.*educational.*equality.*enclave",

    # Biography phrases
    r"israel.*armstead",
    r"israel.*lee",
    r"elvia",
    r"e5 enclave",
    r"farmblock",
    r"mccartney",
    r"black dragons",
    r"restitution 246",
    r"bdi",
    r"west\|abdullah",
    r"cornel west",
    r"national director of finance",

    # Communication style markers
    r"please note that",
    r"for expediency sake",
    r"for accuracy sake",
    r"in the hope that my voice",

    # Dominant close
    r"you're welcome",
    r"please find attached",
]

# Phrases that indicate NOT the Chairman (other speakers, filler, etc.)
NOT_CHAIRMAN_PHRASES = [
    r"^(hey|hi|hello|yo|sup|what's up)",
    r"^(okay cool|alright|ok so)",
    r"(uh+|um+|like,+ you know)",
    r"^(so anyway|anyway so)",
    r"(haha|lmao|lol)\s*$",
]


def score_transcript(text: str) -> dict:
    """
    Score a transcript against the Chairman's voice profile.
    Returns score (0-1) and breakdown.
    """
    text_lower = text.lower()
    text_clean = re.sub(r'[^\w\s]', ' ', text_lower).strip()

    if not text_clean or len(text_clean) < 10:
        return {"score": 0.0, "matches": [], "not_chairman": []}

    matches = []
    for pattern in CHAIRMAN_PHRASES:
        if re.search(pattern, text_lower) or re.search(pattern, text_clean):
            matches.append(pattern)

    not_chairman = []
    for pattern in NOT_CHAIRMAN_PHRASES:
        if re.search(pattern, text_lower):
            not_chairman.append(pattern)

    # Calculate score
    if not_chairman:
        penalty = len(not_chairman) * 0.15
    else:
        penalty = 0

    # Base score from matches (each match adds weight)
    raw_score = min(1.0, len(matches) * 0.12)

    # Length normalization — Chairman tends to speak in longer, structured sentences
    word_count = len(text_clean.split())
    if word_count > 50:
        length_bonus = 0.1
    elif word_count > 20:
        length_bonus = 0.05
    else:
        length_bonus = 0

    total_score = min(1.0, max(0.0, raw_score + length_bonus - penalty))

    return {
        "score": round(total_score, 3),
        "matches": matches,
        "not_chairman": not_chairman,
        "word_count": word_count,
    }


def transcribe_and_score(wav_path: str, model) -> dict:
    """Transcribe a single WAV file and score it."""
    try:
        result = model.transcribe(wav_path, language="en", fp16=False)
        text = result["text"].strip()
        score_data = score_transcript(text)
        return {
            "file": wav_path,
            "text": text,
            **score_data,
            "error": None,
        }
    except Exception as e:
        return {
            "file": wav_path,
            "text": "",
            "score": 0.0,
            "matches": [],
            "not_chairman": [],
            "word_count": 0,
            "error": str(e),
        }


def batch_transcribe(segment_dir: str, model_name: str = "tiny") -> list:
    """Transcribe all WAV files in the segments directory."""
    wav_files = sorted(Path(segment_dir).glob("*.wav"))
    print(f"  Found {len(wav_files)} WAV files")

    print(f"  Loading Whisper model: {model_name}...")
    model = whisper.load_model(model_name)
    print(f"  Model loaded. Beginning transcription...")

    results = []
    total = len(wav_files)

    for i, wav_file in enumerate(wav_files):
        if i % 50 == 0 and i > 0:
            elapsed_total = i  # approximate
            pct = (i / total) * 100
            print(f"    [{i}/{total}] {pct:.0f}% ...")

        result = transcribe_and_score(str(wav_file), model)
        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments-dir", required=True)
    parser.add_argument("--voice-profile", required=True)
    parser.add_argument("--output-dir", default="./voice_data/israel_transcribed")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Min score to classify as Chairman")
    parser.add_argument("--model", default="tiny", choices=["tiny", "base", "small"])
    args = parser.parse_args()

    segment_dir = Path(args.segments_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🔊 TRANSCRIBE + SPEECH PATTERN MATCHING")
    print("=" * 60)
    print(f"  Segments: {segment_dir}")
    print(f"  Model: Whisper {args.model}")
    print(f"  Threshold: {args.threshold}")
    print(f"  Time: {datetime.now().isoformat()}")

    # Phase 1: Transcribe
    print("\n📝 Phase 1: Transcribing all segments...")
    results = batch_transcribe(str(segment_dir), args.model)

    # Phase 2: Classify
    print("\n🎯 Phase 2: Classifying speakers...")
    chairman_segments = []
    other_segments = []
    errors = []

    for r in results:
        if r["error"]:
            errors.append(r)
        elif r["score"] >= args.threshold:
            chairman_segments.append(r)
        else:
            other_segments.append(r)

    # Phase 3: Copy Chairman segments
    print(f"\n📂 Phase 3: Isolating {len(chairman_segments)} Chairman segments...")
    chairman_dir = output_dir / "chairman"
    chairman_dir.mkdir(exist_ok=True)

    for r in chairman_segments:
        src = Path(r["file"])
        dest = chairman_dir / src.name
        shutil.copy2(src, dest)

    # Phase 4: Save transcript + metadata
    transcript_path = output_dir / "transcripts.json"
    with open(transcript_path, "w") as f:
        json.dump({
            "created": datetime.now().isoformat(),
            "threshold": args.threshold,
            "model": args.model,
            "total_segments": len(results),
            "chairman_count": len(chairman_segments),
            "other_count": len(other_segments),
            "error_count": len(errors),
            "results": results,
        }, f, indent=2, default=str)

    # Summary
    print("\n" + "=" * 60)
    print("📊 RESULTS")
    print("=" * 60)
    print(f"  Total segments processed: {len(results)}")
    print(f"  Chairman segments KEPT: {len(chairman_segments)}")
    print(f"  Other speakers REJECTED: {len(other_segments)}")
    print(f"  Errors: {len(errors)}")
    print(f"  Accuracy estimate: based on {len(CHAIRMAN_PHRASES)} voice patterns")
    print(f"  Output: {output_dir}")
    print(f"  Transcript: {transcript_path}")
    print(f"  Files on disk: {len(list(chairman_dir.glob('*.wav')))}")
    print("=" * 60)

    # Show top matches
    print("\n🏆 TOP 10 CHAIRMAN SEGMENTS (highest confidence):")
    sorted_chairman = sorted(chairman_segments, key=lambda x: x["score"], reverse=True)
    for r in sorted_chairman[:10]:
        fname = Path(r["file"]).name
        text_preview = r["text"][:80].replace("\n", " ")
        print(f"  [{r['score']:.2f}] {fname}: \"{text_preview}\"")

    print("\n❌ TOP 10 REJECTED (highest scoring non-Chairman):")
    sorted_other = sorted(other_segments, key=lambda x: x["score"], reverse=True)
    for r in sorted_other[:10]:
        fname = Path(r["file"]).name
        text_preview = r["text"][:80].replace("\n", " ")
        print(f"  [{r['score']:.2f}] {fname}: \"{text_preview}\"")


if __name__ == "__main__":
    main()
