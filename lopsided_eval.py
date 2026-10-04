#!/usr/bin/env python3
"""
Reproduce the ORIGINAL (lopsided) evaluation that the balanced run replaced,
for the README's "why balanced sampling" explanation.

Binary key (as used in early steps): rating >= 4 -> positive, else negative.
Batches: first 100 rows in file order; random 100 (seed 42).
Also reports the naive baseline: % of the cohort that is positive-class
(i.e., what "always predict positive" would score).
"""
import json
import gzip
import random

SRC = "/Users/preethiselvaganapathy/Downloads/AI & LLM/Gift_Cards_classified_with_neutral.jsonl.gz"

with gzip.open(SRC, "rt", encoding="utf-8") as f:
    rows = [json.loads(l) for l in f]

def truth(r):
    return "positive" if float(r["rating"]) >= 4 else "negative"


def report(batch, name):
    n = len(batch)
    agree = sum(1 for r in batch if r["sentiment"] == truth(r))
    n_pos = sum(1 for r in batch if truth(r) == "positive")
    n_neg = n - n_pos
    caught_neg = sum(1 for r in batch
                     if r["sentiment"] == "negative" and truth(r) == "negative")
    print(f"[{name}] {n} reviews")
    print(f"  agreement            : {agree}/{n} = {100.0*agree/n:.1f}%")
    print(f"  positive-class share : {n_pos}/{n} = {100.0*n_pos/n:.1f}%  "
          f"(what 'always say positive' would score)")
    print(f"  true negatives       : {n_neg}  -> caught {caught_neg} "
          f"({100.0*caught_neg/max(n_neg,1):.1f}%)  <- tiny denominator")


print("ORIGINAL LOPSIDED EVALUATION (binary key >=4 positive, else negative)")
print()
report(rows[:100], "first 100 rows in file")
random.seed(42)
report(random.sample(rows, 100), "random 100 (seed 42)")
