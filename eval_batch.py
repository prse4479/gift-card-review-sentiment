#!/usr/bin/env python3
"""
Evaluate the text-only classifier against the 3-class rating-derived key.

Answer key (worked out from the rating, AFTER classification — the model
never sees the rating):
    rating 4-5  ->  POSITIVE
    rating 3    ->  NEUTRAL
    rating 1-2  ->  NEGATIVE

Evaluation set: a BALANCED sample of ~50 rows per class pulled from the
whole file with a fixed seed (2026) — so the rare classes (1-2 star and
3 star) are represented equally instead of drowned out by 5-star text.

Usage: python3 eval_batch.py
"""
import json
import gzip

SRC = "/Users/preethiselvaganapathy/Downloads/AI & LLM/sample_balanced.jsonl.gz"
CLASSES = ["positive", "neutral", "negative"]


def truth_from_rating(rating):
    if rating >= 4:
        return "positive"
    if rating == 3:
        return "neutral"
    return "negative"


def cluster_of(pred, truth):
    if pred == truth:
        return None  # correct
    if pred == "neutral":
        return "abstained"            # model found no signal, key says pos/neg
    if pred == "positive":
        return "over_positive"        # model positive, key neutral/negative
    return "over_negative"            # model negative, key neutral/positive


def main():
    with gzip.open(SRC, "rt", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]

    total = len(rows)
    agree = 0
    conf = {p: {t: 0 for t in CLASSES} for p in CLASSES}
    per_pred = {p: [0, 0] for p in CLASSES}
    per_truth = {t: [0, 0] for t in CLASSES}
    clusters = {"abstained": 0, "over_positive": 0, "over_negative": 0}
    wrong = []

    for r in rows:
        truth = r["key"]
        pred = r["our_sentiment"]
        ok = pred == truth
        agree += ok
        conf[pred][truth] += 1
        per_pred[pred][0] += 1
        per_pred[pred][1] += ok
        per_truth[truth][0] += 1
        per_truth[truth][1] += ok
        cl = cluster_of(pred, truth)
        if cl:
            clusters[cl] += 1
            wrong.append((r["rating"], truth, pred, r["our_score"],
                          r["title"], r["text"]))

    print(f"balanced sample: {total} reviews (50 per class, seed 2026)")
    print(f"AGREEMENT with 3-class key: {agree}/{total} = "
          f"{100.0*agree/total:.1f}%")
    print()
    print("confusion (rows=predicted, cols=answer key):")
    print("pred\\key   " + "".join(f"{t:>10s}" for t in CLASSES))
    for p in CLASSES:
        print(f"{p:10s} " + "".join(f"{conf[p][t]:10d}" for t in CLASSES))
    print()
    print("precision (right when the model says X):")
    for p in CLASSES:
        n, c = per_pred[p]
        print(f"  {p:8s}: {c}/{n} = {100.0*c/n:.1f}%")
    print("recall (caught of true class X):")
    for t in CLASSES:
        n, c = per_truth[t]
        print(f"  {t:8s}: {c}/{n} = {100.0*c/n:.1f}%")
    print()
    print("wrong:", len(wrong), "-> clusters:", clusters)
    print("example wrong rows:")
    for w in wrong[:14]:
        print(f"  [{w[1]:8s}|{w[2]:8s}|{w[0]:.1f}★|{w[3]:+.3f}] "
              f"{w[4][:36]:38s} {w[5][:50]}")


if __name__ == "__main__":
    main()
