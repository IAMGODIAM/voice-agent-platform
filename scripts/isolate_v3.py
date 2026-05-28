#!/usr/bin/env python3
"""
Voice Isolation v3 — Chairman's Voice Sample + V3 Pattern Matching
==================================================================
Uses two signals:
1. Voice pattern matching: exact phrases from Chairman's corpus (Pattern 15, etc.)
2. Semantic similarity: embedding match against Chairman's voice anchors

Combined scoring gives higher accuracy than either method alone.

Usage:
    python3 isolate_v3.py --transcripts transcripts.json --output-dir ./isolated
"""

import json
import shutil
import argparse
import re
import numpy as np
from pathlib import Path
from datetime import datetime

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ── Chairman's Voice Patterns (exact phrase matching) ──────────────────
# These are distinctive phrases that ONLY the Chairman would say verbatim.
# Weighted by uniqueness — Pattern 15 (Directive Formula) is the highest.

PATTERN_WEIGHTS = {
    # Pattern 15 — THE DIRECTIVE FORMULA (highest weight — unique to Chairman)
    r"\bincumbent on you\b": 3.0,
    r"\bmeticulous attention to detail\b": 3.0,
    r"\bmasterfully execute\b": 2.5,
    r"\bexpert masterpiece\b": 2.5,
    r"\bproduce ready work\b": 2.0,
    r"\breview.*assess.*rate.*refine\b": 2.5,

    # Voice sample from 2026-05-28 (Chairman's direct speech)
    r"\bbest at getting better\b": 2.0,
    r"\bgetting better at getting better\b": 3.0,
    r"\bmove all what it up\b": 2.0,
    r"\bemphasis.*excellence\b": 1.5,
    r"\btowards perfection\b": 1.5,
    r"\bmove the people with passion\b": 2.0,
    r"\bparsnickety\b": 3.0,
    r"\baudio wrangling\b": 2.0,

    # Pattern 6 — Latin Mottos
    r"\bnil satis nisi optimum\b": 3.0,
    r"\bdum spiro\b": 2.0,
    r"\bwhile i breathe.*hope\b": 2.0,

    # Pattern 13 — Seed Metaphor
    r"\bseed.*planted\b": 1.5,
    r"\bit was a seed\b": 2.0,

    # Pattern 10 — Already-Invested Authority
    r"\bi own.*acres?\b": 2.0,
    r"\bi bought it.*150\b": 2.5,
    r"\bthat is not a metaphor\b": 3.0,
    r"\bpine bluff.*arkansas\b": 1.5,

    # Pattern 11 — Prayer Register
    r"\bthe center holds\b": 3.0,
    r"\bthe work is underway\b": 2.0,
    r"\bchildren of god\b": 1.5,

    # Pattern 9 — Civilizational Pivot
    r"\bliberty city.*food desert\b": 2.5,
    r"\bground-level expression\b": 2.0,

    # Project Lexicon (distinctive terms)
    r"\be5 enclave\b": 1.0,
    r"\bfarmblock\b": 1.0,
    r"\bmccartney.*academy\b": 1.0,
    r"\brestitution 246\b": 1.5,
    r"\bblack dragons?\b": 1.0,
    r"\bwest.*abdullah\b": 1.5,
    r"\bnational director of finance\b": 2.0,
    r"\bIAMGODIAM\b": 1.5,
}

# Semantic anchors for embedding matching
SEMANTIC_ANCHORS = [
    "It is incumbent on you to approach this task step-by-step while paying meticulous attention to detail.",
    "We expect you to masterfully execute an expert masterpiece and produce ready work.",
    "Review, Assess, Rate, and Refine.",
    "Getting good at getting better with emphasis and excellence in mind towards perfection.",
    "Speak in the register that is necessary to move the people with passion.",
    "The lineage built this country. We are building what the lineage is owed.",
    "I own 1.2 acres in Pine Bluff Arkansas. That is not a metaphor. That is a deed.",
    "Liberty City's food desert is not a local accident.",
    "Nil satis nisi optimum. Dum spiro, spero.",
    "We are always black first.",
]


def score_keyword(text: str) -> tuple:
    """Score text against keyword patterns. Returns (score, matched_patterns)."""
    text_lower = text.lower()
    score = 0.0
    matched = []
    for pattern, weight in PATTERN_WEIGHTS.items():
        if re.search(pattern, text_lower):
            score += weight
            matched.append(pattern)
    return score, matched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcripts", required=True)
    parser.add_argument("--output-dir", default="./voice_data/isolated_v3")
    parser.add_argument("--min-words", type=int, default=5)
    parser.add_argument("--kw-threshold", type=float, default=2.0,
                        help="Min keyword score to classify as Chairman")
    parser.add_argument("--emb-threshold", type=float, default=0.30,
                        help="Min embedding similarity threshold")
    parser.add_argument("--strategy", default="combined",
                        choices=["keyword", "embedding", "combined"])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("══════════════════════════════════════════════")
    print("🔊 VOICE ISOLATION v3 — Keyword + Semantic")
    print(f"  Strategy: {args.strategy}")
    print(f"  KW threshold: {args.kw_threshold}")
    print(f"  EMB threshold: {args.emb_threshold}")

    # Load transcripts
    with open(args.transcripts) as f:
        results = json.load(f)["results"]

    print(f"\n📝 Loaded {len(results)} transcripts")

    # ── Phase 1: Keyword Scoring ──
    print("\n🔑 Phase 1: Keyword pattern matching...")
    for r in results:
        if r.get("error") or not r.get("text"):
            r["kw_score"] = 0.0
            r["kw_matches"] = []
        else:
            score, matched = score_keyword(r.get("text", ""))
            r["kw_score"] = score
            r["kw_matches"] = matched

    # Stats
    kw_positive = sum(1 for r in results if r.get("kw_score", 0) >= args.kw_threshold)
    print(f"  Pattern matches (>= {args.kw_threshold}): {kw_positive}")

    # ── Phase 2: Embedding scoring ──
    print("\n🧠 Phase 2: Semantic embedding matching...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    anchor_embs = model.encode(SEMANTIC_ANCHORS, show_progress_bar=False, convert_to_numpy=True)
    anchor_mean = np.mean(anchor_embs, axis=0)

    valid_results = [r for r in results
                     if not r.get("error")
                     and len(r.get("text", "").strip().split()) >= args.min_words]

    batch_size = 128
    all_texts = [r.get("text", "").strip() for r in valid_results]
    all_embs = []

    for i in range(0, len(all_texts), batch_size):
        batch = all_texts[i:i+batch_size]
        embs = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        all_embs.extend(embs)

    similarities = cosine_similarity(all_embs, anchor_mean.reshape(1, -1)).flatten()

    for i, r in enumerate(valid_results):
        r["emb_score"] = float(similarities[i])

    # ── Phase 3: Combined Classification ──
    print(f"\n🎯 Phase 3: Classification (strategy: {args.strategy})...")
    chairman_dir = output_dir / "chairman"
    chairman_dir.mkdir(exist_ok=True)

    chairman_count = 0
    other_count = 0
    skipped = 0
    transcript_data = []

    for r in results:
        if r.get("error"):
            skipped += 1
            continue

        text = r.get("text", "").strip()
        words = text.split()
        if len(words) < args.min_words:
            skipped += 1
            continue

        kw = r.get("kw_score", 0.0)
        emb = r.get("emb_score", 0.0)

        # Combined score: keyword is binary signal (presence of unique phrases),
        # embedding is gradational (overall style similarity)
        if args.strategy == "keyword":
            is_chairman = kw >= args.kw_threshold
        elif args.strategy == "embedding":
            is_chairman = emb >= args.emb_threshold
        else:  # combined
            # If keyword matches found → strong signal
            if kw >= args.kw_threshold:
                is_chairman = True
            elif kw >= 1.0 and emb >= args.emb_threshold * 0.8:
                # Borderline keyword + decent embedding
                is_chairman = True
            elif emb >= args.emb_threshold + 0.15:
                # Very strong embedding match alone
                is_chairman = True
            else:
                is_chairman = False

        entry = {
            "file": r["file"],
            "text": text[:200],
            "kw_score": round(kw, 2),
            "emb_score": round(emb, 4),
            "words": len(words),
            "kw_matches": r.get("kw_matches", [])[:5],
            "classification": "chairman" if is_chairman else "other",
        }
        transcript_data.append(entry)

        if is_chairman:
            shutil.copy2(r["file"], chairman_dir / Path(r["file"]).name)
            chairman_count += 1
        else:
            other_count += 1

    # Save classified data
    with open(output_dir / "classified.json", "w") as f:
        json.dump({
            "created": datetime.now().isoformat(),
            "strategy": args.strategy,
            "kw_threshold": args.kw_threshold,
            "emb_threshold": args.emb_threshold,
            "total": len(results),
            "chairman": chairman_count,
            "other": other_count,
            "skipped": skipped,
            "segments": sorted(transcript_data, key=lambda x: x["kw_score"], reverse=True),
        }, f, indent=2)

    # Summary
    print("\n══════════════════════════════════════════════")
    print("📊 FINAL RESULTS")
    print("══════════════════════════════════════════════")
    print(f"  Strategy:              {args.strategy}")
    print(f"  Chairman KEPT:         {chairman_count} ({100*chairman_count/(chairman_count+other_count):.1f}%)")
    print(f"  Other speakers:        {other_count}")
    print(f"  Skipped (short/error): {skipped}")
    print(f"  Files on disk:         {len(list(chairman_dir.glob('*.wav')))}")

    # Top matches
    chairman_entries = [e for e in transcript_data if e["classification"] == "chairman"]
    chairman_entries.sort(key=lambda x: x["kw_score"], reverse=True)
    print(f"\n🏆 TOP 10 CHAIRMAN (by keyword score):")
    for e in chairman_entries[:10]:
        text = e["text"][:80].replace("\n", " ")
        matches = ", ".join(e["kw_matches"][:3]) if e["kw_matches"] else "—"
        print(f"  [kw={e['kw_score']:.1f} emb={e['emb_score']:.3f}] \"{text}\"")
        if matches != "—":
            print(f"    matches: {matches}")

    print(f"\n✅ Output: {chairman_dir}")
    print("══════════════════════════════════════════════")


if __name__ == "__main__":
    main()
