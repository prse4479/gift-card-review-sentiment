#!/usr/bin/env python3
"""
Step 6 — build the balanced 3-class sample.

Answer key (3-class):
    rating 4-5 -> POSITIVE
    rating 3   -> NEUTRAL
    rating 1-2 -> NEGATIVE

Balanced sampling: ~50 rows per class, picked with a fixed seed from the
WHOLE file so the same set comes up every run. Rows carry the answer-key
class, our text classifier's label, and NRC emotions for later steps.
"""
import json
import gzip
import random

CLASSES = [("positive", "4-5 star"), ("neutral", "3 star"), ("negative", "1-2 star")]
SEED = 2026
PER_CLASS = 50

EMOTIONS_SRC = "/Users/preethiselvaganapathy/Downloads/AI & LLM/Gift_Cards_classified_emotions.jsonl.gz"
OUT = "/Users/preethiselvaganapathy/Downloads/AI & LLM/sample_balanced.jsonl.gz"


def key_class(rating):
    if rating >= 4:
        return "positive"
    if rating == 3:
        return "neutral"
    return "negative"


def main():
    with gzip.open(EMOTIONS_SRC, "rt", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]
    total = len(rows)

    buckets = {c: [] for c, _ in CLASSES}
    for i, r in enumerate(rows):
        rating = float(r["rating"])
        buckets[key_class(rating)].append(i)

    print("class sizes in full dataset:")
    for c, label in CLASSES:
        print(f"  {c:9s} ({label:9s}): {len(buckets[c])} ({100.0*len(buckets[c])/total:.2f}%)")

    rng = random.Random(SEED)
    picked = []
    for c, _ in CLASSES:
        pool = buckets[c]
        take = min(PER_CLASS, len(pool))
        picked.extend(rng.sample(pool, take))
        print(f"  picked {take} from {c}")
    picked.sort()

    sample = []
    for i in picked:
        r = rows[i]
        sample.append({
            "idx": i,
            "rating": float(r["rating"]),
            "key": key_class(float(r["rating"])),
            "our_sentiment": r["sentiment"],
            "our_score": r["sentiment_score"],
            "nrc_emotion": r["nrc_emotion"],
            "title": r.get("title") or "",
            "text": r.get("text") or "",
        })

    with gzip.open(OUT, "wt", encoding="utf-8") as f:
        for s in sample:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"sample size: {len(sample)} -> written {OUT}")


if __name__ == "__main__":
    main()
