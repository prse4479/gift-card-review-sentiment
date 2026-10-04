#!/usr/bin/env python3
"""
Amazon Gift Card review sentiment classifier (Python), 3-class.

Reads:  Gift_Cards.jsonl.gz  (Amazon review JSONL: title, text, ...)
Writes: Gift_Cards_classified_with_neutral.jsonl.gz
        (original fields + sentiment [positive|negative|neutral] + sentiment_score)

Method:
  1. Phrase pre-check on title + text (lowercased), BEFORE VADER:
       - complaint phrases            (e.g. "does not work", "ripped")  -> negative
       - rhetorical-positive phrases  (e.g. "can't go wrong")            -> positive
       - factual-positive phrases     (e.g. "delivered on time", "works")-> positive
     Complaints win over positives. The lists catch wording a sentiment
     lexicon cannot see: verbs, idioms, misspellings, rhetorical questions,
     and praise with no sentiment words at all.
  2. Otherwise VADER lexicon sentiment -> compound score in [-1, 1]
       compound >  0  ->  positive
       compound <  0  ->  negative
       compound == 0  ->  neutral   (text has no sentiment words at all)
  3. Star rating is not read or used in any way.

Usage:
    python3 classify_gift_cards.py [input.jsonl.gz]
    python3 classify_gift_cards.py [input.jsonl.gz] [output.jsonl.gz]
"""
import json
import gzip
import sys
from collections import Counter

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    sys.exit("Missing dependency. Install with:  pip3 install vaderSentiment")

SRC_DEFAULT = "/Users/preethiselvaganapathy/Downloads/AI & LLM/Gift_Cards.jsonl.gz"
OUT_DEFAULT = "/Users/preethiselvaganapathy/Downloads/AI & LLM/Gift_Cards_classified_with_neutral.jsonl.gz"

# Substrings checked on lowercased title+text. COMPLAINTS win over the
# positive lists so mixed reviews ("arrived on time but card is torn")
# go negative. Factual-positive phrases catch praise that has no
# sentiment words ("delivered on time and as described").
COMPLAINT_PHRASES = [
    # failure / activation
    "does not work", "doesn't work", "dont work", "do not work", "don't work",
    "did not work", "didn't work", "not working", "stopped working", "never worked",
    "won't work", "wont work", "wouldn't work", "would not work",
    "does not activate", "not activated", "not active", "never activated",
    "won't activate", "wont activate", "didn't activate", "never activate",
    # delivery / receipt
    "never received", "did not receive", "didn't receive", "never got", "never got it",
    "never arrived", "did not arrive", "didn't arrive", "never came", "never showed",
    "never showed up", "only received", "not received",
    "didn't arrive", "did not arrive", "never showed up",
    "not on time", "wasn't on time", "was not on time", "arrived late", "came late",
    "delivered late",
    "wrong address", "wrong item", "wrong card", "wrong order", "wrong house",
    "not as expected", "wasn't as expected", "was not as expected",
    "not what i expected", "wasn't what i expected", "not what was expected",
    "not as described", "wasn't as described", "not as advertised",
    "not as shown", "not as pictured", "not as listed",
    # physical / condition
    "ripped", "torn", "shredded", "damaged", "broken", "bent", "crushed",
    "too small", "not valid", "invalid", "expired",
    # count / fraud
    "shorted", "missing", "scam", "fraud", "fake", "counterfeit", "compromised",
    "deactivate", "deactivated", "desactivate",
    # purchase-regret / confusion
    "don't buy", "do not buy", "dont buy",
    "can't figure out", "cant figure out", "cannot figure out",
    # sentiment words VADER misses (incl. common misspellings)
    "useless", "junk", "waste of money", "hassle", "nightmare", "refund",
    "not satisfied", "not satisffied", "unsatisfied", "not pleased", "not happy",
    "dissapoint",   # catches misspelled "disappointed"
    "frustrat",     # catches "frustrated" / "frustrating"
]

RHETORIC_POSITIVE_PHRASES = [
    "what's not to like", "what is not to like", "whats not to like",
    "what's not to love", "what is not to love", "whats not to love",
    "can't go wrong", "cant go wrong", "cannot go wrong", "can't go too wrong",
    "can't beat", "cant beat", "can't be beat", "cant be beat", "hard to beat",
    "no complaints", "can't complain", "cant complain", "nothing to complain",
    "no issues", "no problems", "no problem", "no trouble", "without issue",
    "who doesn't love", "who doesnt love", "who wouldn't love", "who wouldnt love",
    "who doesn't like", "who doesnt like", "who wouldn't want", "who wouldnt want",
    "who doesn't want", "who doesnt want",
]

# Praise expressed with factual words only (no sentiment words for VADER).
# Complaint list is checked first, so negatives like "not on time" win.
FACTUAL_POSITIVE_PHRASES = [
    "as described", "as advertised", "just as described", "exactly as described",
    "exactly as advertised", "as pictured", "as shown", "as listed",
    "as expected", "just as expected", "exactly as expected",
    "on time", "arrived", "delivered", "received",
    "quick delivery", "fast delivery", "fast shipping", "quick shipping",
    "shipped fast", "shipped quickly", "came quickly", "came fast", "arrived fast",
    "works", "worked", "it works", "they work", "work fine", "works fine",
    "worked fine", "works as expected", "worked as expected",
    "works perfectly", "worked perfectly", "everything works", "everything worked",
    "just what i needed", "just what i wanted", "exactly what i needed",
    "exactly what i wanted",
]


def precheck(combined_lower):
    """Phrase rules that run before VADER. Returns a label or None."""
    for p in COMPLAINT_PHRASES:
        if p in combined_lower:
            return "negative"
    for p in RHETORIC_POSITIVE_PHRASES + FACTUAL_POSITIVE_PHRASES:
        if p in combined_lower:
            return "positive"
    return None


def normalize_quotes(text):
    """Replace curly quotes/apostrophes with straight ones.

    Phrase lists are written with straight ' apostrophes, but many reviews
    use the curly character ('Didn't work'). Without this, "didn't work"
    never matches and the review is mislabeled. Output text is unchanged;
    only the matching copy is normalized.
    """
    for curly, straight in [("\u2019", "'"), ("\u2018", "'"),
                            ("\u201c", '"'), ("\u201d", '"')]:
        text = text.replace(curly, straight)
    return text


def classify(analyzer, title, text):
    """Return (label, compound_score) using pre-check + VADER."""
    combined = f"{title or ''} {text or ''}".strip()
    if not combined:
        return "neutral", 0.0  # no text at all -> no signal
    combined = normalize_quotes(combined)
    low = combined.lower()
    label = precheck(low)
    score = analyzer.polarity_scores(combined)["compound"]
    if label:
        return label, score
    if score > 0:
        return "positive", score
    if score < 0:
        return "negative", score
    return "neutral", 0.0  # compound == 0: no sentiment words in text


def main(src=SRC_DEFAULT, out=OUT_DEFAULT):
    analyzer = SentimentIntensityAnalyzer()
    counts = Counter()
    precheck_hits = Counter()
    total = 0

    fin = gzip.open(src, "rt", encoding="utf-8") if src.endswith((".gz", ".gzip")) \
        else open(src, encoding="utf-8")
    with fin as f, gzip.open(out, "wt", encoding="utf-8") as fout:
        for line in f:
            d = json.loads(line)
            total += 1
            d["title"] = d.get("title") or ""
            d["text"] = d.get("text") or ""
            label, score = classify(analyzer, d["title"], d["text"])

            d["sentiment"] = label
            d["sentiment_score"] = round(score, 4)
            fout.write(json.dumps(d) + "\n")
            counts[label] += 1
            if precheck(normalize_quotes(f"{d['title']} {d['text']}").lower()):
                precheck_hits[label] += 1

    print(f"total: {total}")
    print(f"labels: {dict(counts)}")
    print(f"percent: { {k: round(100.0*v/total, 2) for k, v in counts.items()} }")
    print(f"pre-check phrases fired: {dict(precheck_hits)} — "
          f"{round(100.0*sum(precheck_hits.values())/total, 2)}% of all reviews")
    print(f"output written to: {out}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
