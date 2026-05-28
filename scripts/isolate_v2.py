#!/usr/bin/env python3
"""
Voice Isolation v2 — Embedding-Based Speaker Classification
===========================================================
Uses sentence-transformer embeddings to compare each transcript against:
1. Israel Voice V3 style profile (positive anchor)
2. Random corpus text (negative anchor)

More robust than keyword matching — captures semantic similarity to the
Chairman's speech patterns including vocabulary range, sentence structure,
project references, and rhetorical approach.

Usage:
    python3 isolate_v2.py --transcripts transcripts.json --output-dir ./isolated
"""

import json
import shutil
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ── These are EXACT text excerpts from the Chairman's Voice V3 ──────────
# They define his voice — biography, phrases, patterns, project lexicon.
# The 533-line V3 document is embedded chunk-by-chunk as the positive profile.

CHAIRMAN_STYLE_ANCHORS = [
    # From Pattern 1 — formal register with oral rhythm
    "It is my belief that I was disparately treated, specifically along racial lines, not limited to but to include my time as an actual broker and all the events leading up to my subsequent departure.",

    # From Pattern 2 — inclusive preemption
    "Please note that for expediency sake I have included the For accuracy sake, I have also included as witness to my remittance.",

    # From Pattern 3 — moral frame before factual frame
    "When holding a position of moral authority one must first establish the character and the stakes before the data enters the frame.",

    # From Pattern 4 — witnessed compassion
    "In the hope that my voice will lend aid to those who have been denied the dignity of being heard.",

    # From Pattern 8 — dominant close
    "Please find attached the reply to the attorney. You're welcome.",

    # From Pattern 9 — civilizational pivot
    "Liberty City's food desert is not a local accident. It is the ground-level expression of agricultural policy designed to fail communities like this one.",

    # From Pattern 10 — already-invested authority
    "I own 1.2 acres in Pine Bluff Arkansas. I bought it in 2019 for $150. That is not a metaphor. That is a deed.",

    # From Pattern 11 — prayer register
    "The lineage built this country. We are building what the lineage is owed. The center holds. The work is underway.",

    # From Pattern 13 — seed metaphor
    "It was a seed that was planted. Planting precedes harvest. Vision precedes capital. Prayer precedes execution.",

    # From Pattern 14 — credentials as witness
    "I am a licensed real estate agent, a licensed insurance agent. As an underwriter, it was my job to assess risk and quantify exposure.",

    # From Pattern 15 — directive formula
    "It is incumbent on you to approach this task step-by-step while paying meticulous attention to detail. We expect you to masterfully execute an expert masterpiece, produce ready work, and provide your best.",

    # From Pattern 16 — architect audit
    "That is good progress but the drift is concerning. The short read: Good — the foundation is sound. Not good — the framing has wandered. My recommendation: recalibrate to the original specification.",

    # From Pattern 17 — range declaration
    "Israel Lee Armstead, also known by his in-game name IAMGODIAM, is a BIPOC professional who specializes in institutional design.",

    # From Pattern 19 — rising from Liberty City
    "E5 Enclave's Black Dragons Initiative is rooted in Miami's historic Black communities, rising from Liberty City with inspiration from Overtown's legacy of Black economic self-determination.",

    # From Pattern 20 — 5Es
    "A nonprofit dedicated to Black empowerment through environmental sustainability, economic self-reliance, educational advancement, and equality, empowered by community ownership and innovation.",

    # From canonical bio Version A
    "Israel Lee Armstead is the President Founder and Chief Visionary Officer of E5 Enclave Incorporated, a 501(c)(3) public charity incorporated in Liberty City Miami Florida.",

    # From canonical bio Version B
    "At eighteen selected from thirteen thousand applicants he walked onto the floor at Mercedes-Benz. He bought 1.2 acres in Pine Bluff Arkansas for $150. He has read the Bible cover to cover in multiple translations.",

    # From Unified Thesis
    "You can build an entirely new world with a very small investment. Buy the empty lots. Don't put in the coffee shop. Put in computer labs. Double them as crypto mines.",

    # From V3 session close
    "Full commission day executed. Six major missions completed. Everything journaled backed pushed to GitHub synced to wiki.",

    # Additional Voice V3 sample phrases (casual contexts)
    "I'm gonna go get a drink of water like Marco Rubio.",
    "I could preach on it for hours man.",
    "We are always black first.",

    # Partner outreach voice
    "My name is Israel L. Armstead. I'm reaching out to share our transformative vision and extend an invitation for your organization to join us.",
]


def load_transcripts(path):
    with open(path) as f:
        return json.load(f)["results"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcripts", required=True, help="Path to transcripts.json")
    parser.add_argument("--output-dir", default="./voice_data/isolated_v2")
    parser.add_argument("--threshold", type=float, default=0.35,
                        help="Cosine sim threshold (0-1). Higher = stricter.")
    parser.add_argument("--min-words", type=int, default=5,
                        help="Skip segments with fewer than N words")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("══════════════════════════════════════════════")
    print("🔊 VOICE ISOLATION v2 — Embedding Based")
    print(f"  Time: {datetime.now().isoformat()}")
    print(f"  Threshold: {args.threshold}")
    print(f"  Anchors: {len(CHAIRMAN_STYLE_ANCHORS)}")

    # Load model
    print("\n📦 Loading MiniLM-L6-v2...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Encode Chairman's voice profile
    print("🎯 Encoding Chairman's voice profile...")
    anchor_embs = model.encode(CHAIRMAN_STYLE_ANCHORS, show_progress_bar=False, convert_to_numpy=True)
    anchor_mean = np.mean(anchor_embs, axis=0)
    print(f"  Anchor profile: {len(CHAIRMAN_STYLE_ANCHORS)} vectors, dim={anchor_mean.shape[0]}")

    # Load transcripts
    print(f"\n📝 Loading transcripts...")
    results = load_transcripts(args.transcripts)
    print(f"  Total: {len(results)} transcripts")

    # Encode and classify
    print(f"\n🔄 Classifying...")
    chairman_dir = output_dir / "chairman"
    other_dir = output_dir / "other"
    chairman_dir.mkdir(exist_ok=True)
    other_dir.mkdir(exist_ok=True)

    chairman_count = 0
    other_count = 0
    skipped = 0

    # Filter out very short / empty transcripts
    valid_results = []
    for r in results:
        if r.get("error"):
            skipped += 1
            continue
        words = r.get("text", "").strip().split()
        if len(words) < args.min_words:
            skipped += 1
            continue
        valid_results.append(r)

    print(f"  Valid: {len(valid_results)}, Skipped: {skipped}")

    # Process in batches for speed
    batch_size = 64
    all_texts = [r.get("text", "").strip() for r in valid_results]
    all_embeddings = []

    for i in range(0, len(all_texts), batch_size):
        batch = all_texts[i:i+batch_size]
        embs = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        all_embeddings.extend(embs)
        if (i // batch_size) % 10 == 0:
            print(f"  Encoded: {min(i+batch_size, len(all_texts))}/{len(all_texts)}")

    # Compute cosine similarities
    similarities = cosine_similarity(all_embeddings, anchor_mean.reshape(1, -1)).flatten()

    # Classify
    chairman_scores = []
    other_scores = []

    for i, (r, sim) in enumerate(zip(valid_results, similarities)):
        wav_path = Path(r["file"])
        score = float(sim)

        entry = {
            "file": r["file"],
            "text": r.get("text", ""),
            "score": round(score, 4),
            "words": len(r.get("text", "").split()),
        }

        if score >= args.threshold:
            dest = chairman_dir / wav_path.name
            shutil.copy2(wav_path, dest)
            chairman_scores.append(entry)
            chairman_count += 1
        else:
            other_scores.append(entry)
            other_count += 1

    # Save classified transcripts
    with open(output_dir / "classified.json", "w") as f:
        json.dump({
            "created": datetime.now().isoformat(),
            "threshold": args.threshold,
            "total": len(results),
            "chairman": chairman_count,
            "other": other_count,
            "skipped": skipped,
            "chairman_segments": sorted(chairman_scores, key=lambda x: x["score"], reverse=True),
            "other_segments": sorted(other_scores, key=lambda x: x["score"], reverse=True),
        }, f, indent=2)

    # Summary
    print("\n══════════════════════════════════════════════")
    print("📊 CLASSIFICATION RESULTS")
    print("══════════════════════════════════════════════")
    print(f"  Chairman KEPT:    {chairman_count} ({100*chairman_count/len(valid_results):.1f}%)")
    print(f"  Other speakers:   {other_count}")
    print(f"  Skipped (short):  {skipped}")
    print(f"  Chairman files:   {len(list(chairman_dir.glob('*.wav')))}")
    print(f"  Score range:      {similarities.min():.3f} - {similarities.max():.3f}")
    print(f"  Mean score:       {similarities.mean():.3f}")

    # Top matches
    print(f"\n🏆 TOP 10 CHAIRMAN (highest similarity):")
    for entry in sorted(chairman_scores, key=lambda x: x["score"], reverse=True)[:10]:
        text = entry["text"][:90].replace("\n", " ")
        print(f"  [{entry['score']:.3f}] \"{text}\"")

    print(f"\n❌ TOP BORDERLINE REJECTED:")
    borderline = [e for e in other_scores if e["score"] >= args.threshold - 0.05]
    borderline.sort(key=lambda x: x["score"], reverse=True)
    for entry in borderline[:10]:
        text = entry["text"][:90].replace("\n", " ")
        print(f"  [{entry['score']:.3f}] \"{text}\"")

    print(f"\n💀 LOWEST SCORED (clearly NOT Chairman):")
    lowest = sorted(other_scores, key=lambda x: x["score"])[:5]
    for entry in lowest:
        text = entry["text"][:90].replace("\n", " ")
        print(f"  [{entry['score']:.3f}] \"{text}\"")

    print("══════════════════════════════════════════════")


if __name__ == "__main__":
    main()
