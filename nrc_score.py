#!/usr/bin/env python3
"""
NRC emotion scorer — Take #2 (word-list derivation, no model calls).

Reads the classified reviews, scores every review's words against the
public NRC Emotion Lexicon v0.92 (8 emotions), and writes a new file
with the derived primary emotion per review.

Scoring: for each word in the review (title+text, lowercased), add its
NRC emotion associations; the primary emotion is the highest total.
Ties break in the canonical order: anger, anticipation, disgust, fear,
joy, sadness, surprise, trust. Reviews with no emotion words at all
get "none".
"""
import json
import gzip
import os
import re
import sys

# Lexicon sits next to this script (vendored copy of NRC v0.92), so the
# script runs from any folder without external downloads.
LEXICON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt")
SRC = "/Users/preethiselvaganapathy/Downloads/AI & LLM/Gift_Cards_classified_with_neutral.jsonl.gz"
OUT = "/Users/preethiselvaganapathy/Downloads/AI & LLM/Gift_Cards_classified_emotions.jsonl.gz"

EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy",
            "sadness", "surprise", "trust"]

TOKEN_RE = re.compile(r"[a-z']+")


def load_lexicon(path):
    """word -> set of emotions it is associated with (flag == 1)."""
    lex = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            word, cat, flag = parts
            if cat not in EMOTIONS or flag != "1":
                continue
            lex.setdefault(word, set()).add(cat)
    return lex


def score_text(lex, text):
    """Return (primary_emotion, scores_dict) for one review's text."""
    totals = {e: 0 for e in EMOTIONS}
    for tok in TOKEN_RE.findall(text.lower()):
        if tok in lex:
            for e in lex[tok]:
                totals[e] += 1
    best = max(totals.values())
    if best == 0:
        return "none", totals
    # first emotion in canonical order wins ties
    primary = next(e for e in EMOTIONS if totals[e] == best)
    return primary, totals


def main():
    lex = load_lexicon(LEXICON)
    print("lexicon words:", len(lex))
    total = 0
    counts = {e: 0 for e in EMOTIONS + ["none"]}
    n_scrubbed = 0

    with gzip.open(SRC, "rt", encoding="utf-8") as f, \
         gzip.open(OUT, "wt", encoding="utf-8") as out:
        for line in f:
            d = json.loads(line)
            total += 1
            text = (d.get("title") or "") + " " + (d.get("text") or "")
            primary, scores = score_text(lex, text)
            d["nrc_emotion"] = primary
            d["nrc_scores"] = scores
            if primary == "none":
                n_scrubbed += 1
            counts[primary] += 1
            out.write(json.dumps(d) + "\n")
            if total % 50000 == 0:
                print("...", total)

    print("total reviews:", total)
    print("emotion distribution:")
    for e in EMOTIONS + ["none"]:
        print(f"  {e:13s} {counts[e]:7d}  ({100.0*counts[e]/total:.2f}%)")
    print("no emotion words:", n_scrubbed)
    print("written:", OUT)


if __name__ == "__main__":
    main()
