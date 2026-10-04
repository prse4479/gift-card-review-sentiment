#!/usr/bin/env python3
"""
Cross-check: numbers scraped from the dashboard DOM (in the browser)
vs numbers recomputed from the saved data files. Fails loudly on any
mismatch.
"""
import json
import gzip

# ---- values scraped from the rendered page -------------------------------
DOM = {
    "metrics": ["54.7%", "98.0%", "62.0%", "4.0%"],
    "donut": "55%",
    "clusters": ["8", "46", "14"],
    "cmCells": ["49", "35", "11", "0", "2", "8", "1", "13", "31"],
    "starcounts": [12326, 1873, 3271, 6692, 128248],
    "starpcts": [8.10, 1.20, 2.10, 4.40, 84.10],
    "splitRight": ["49/50", "2/50", "31/50"],
    "segs": [[49, 0, 1], [35, 2, 13], [11, 8, 31]],
    "filters": [150, 82, 68, 8, 46, 14],
    "tableRows": 20,
    "classbars": {
        "recall": [98.0, 4.0, 62.0],
        "precision": [51.6, 20.0, 68.9],
    },
}

# ---- recompute everything from the saved files ---------------------------
SAMPLE = "/Users/preethiselvaganapathy/Downloads/AI & LLM/sample_balanced.jsonl.gz"
FULL = "/Users/preethiselvaganapathy/Downloads/AI & LLM/Gift_Cards_classified_emotions.jsonl.gz"

with gzip.open(SAMPLE, "rt") as f:
    sample = [json.loads(l) for l in f]
with gzip.open(FULL, "rt") as f:
    full = [json.loads(l) for l in f]

def key(r):
    r = float(r["rating"])
    return "positive" if r >= 4 else ("neutral" if r == 3 else "negative")

fn = len(full)
agree = sum(1 for r in sample if r["our_sentiment"] == r["key"])
n = len(sample)

star_ct = [0] * 5
for r in full:
    star_ct[int(round(float(r["rating"]))) - 1] += 1

C = {"positive": "P", "neutral": "N", "negative": "G"}
LETTERS = ["P", "N", "G"]
def cc(s):
    return C[s]

conf = {p: {t: 0 for t in LETTERS} for p in LETTERS}
by_key = {t: {p: 0 for p in LETTERS} for t in LETTERS}
for r in full:
    conf[cc(r["sentiment"])][cc(key(r))] += 1
for r in sample:
    by_key[cc(r["key"])][cc(r["our_sentiment"])] += 1

truth = {t: sum(conf[p][t] for p in LETTERS) for t in LETTERS}
pred_ = {p: sum(conf[p][t] for t in LETTERS) for p in LETTERS}
recall = {t: 100.0 * conf[t][t] / truth[t] for t in LETTERS}
precision = {p: 100.0 * conf[p][p] / pred_[p] for p in LETTERS}

clusters = {}
for r in sample:
    if r["our_sentiment"] != r["key"]:
        c = "abstained" if r["our_sentiment"] == "neutral" else (
            "over_positive" if r["our_sentiment"] == "positive" else "over_negative")
        clusters[c] = clusters.get(c, 0) + 1

# ---- assertions -----------------------------------------------------------
errors = []

def check(name, got, want):
    if got != want:
        errors.append(f"{name}: page={got} file={want}")

check("metrics", DOM["metrics"][0], f"{100.0*agree/n:.1f}%")
for i, t in enumerate(("positive", "negative", "neutral")):  # page order
    tp = sum(1 for r in sample if r["key"] == t and r["our_sentiment"] == t)
    tot = sum(1 for r in sample if r["key"] == t)
    check(f"metric[{i+1}]", DOM["metrics"][i + 1], f"{100.0*tp/tot:.1f}%")

check("donut", DOM["donut"], f"{round(100.0*agree/n)}%")
check("clusters", [str(clusters.get(c, 0)) for c in
                   ("abstained", "over_positive", "over_negative")], DOM["clusters"])

want_cm = [conf[p][t] for p in ("P", "N", "G") for t in ("P", "N", "G")]  # full-dataset (sanity only)
full_diag = want_cm[0] + want_cm[4] + want_cm[8]
non_diag = sum(want_cm) - full_diag
print(f"full-dataset diagonal (sanity): {full_diag}/{fn} = "
      f"{100.0*full_diag/fn:.1f}% (non-diagonal {non_diag})")

# page now shows the BALANCED-sample confusion + bars: recompute those
sconf = {p: {t: 0 for t in ("P", "N", "G")} for p in ("P", "N", "G")}
s_per_pred = {p: [0, 0] for p in ("P", "N", "G")}
s_per_truth = {t: [0, 0] for t in ("P", "N", "G")}
for r in sample:
    p, t = cc(r["our_sentiment"]), cc(r["key"])
    sconf[p][t] += 1
    s_per_pred[p][0] += 1
    s_per_pred[p][1] += (p == t)
    s_per_truth[t][0] += 1
    s_per_truth[t][1] += (p == t)

want_cm = [sconf[p][t] for p in ("P", "N", "G") for t in ("P", "N", "G")]
check("cmCells", [f"{v:,}" for v in want_cm], DOM["cmCells"])

check("starcounts", star_ct, DOM["starcounts"])
# page renders pct rounded to 1 decimal then formatted to 2 digits
check("starpcts", [f"{round(100.0 * c / fn, 1):.2f}" for c in star_ct],
      [f"{v:.2f}" for v in DOM["starpcts"]])

for i, t in enumerate(("P", "N", "G")):
    s = sum(by_key[t].values())
    right = by_key[t][t]
    check(f"splitRight[{i}]", DOM["splitRight"][i], f"{right}/{s}")
    check(f"segs[{i}]",
          [by_key[t][p] for p in ("P", "N", "G")], DOM["segs"][i])

check("filters[0..1]", [n, agree] + [n - agree] +
      [clusters.get(c, 0) for c in ("abstained", "over_positive", "over_negative")],
      DOM["filters"])
check("tableRows", 20, DOM["tableRows"])  # paginated page shows 20 per page

# full-dataset diagonal (sanity print, no longer on the page)
diag = 0
check("recall", [round(100.0 * s_per_truth[t][1] / s_per_truth[t][0], 1)
                 for t in ("P", "N", "G")], DOM["classbars"]["recall"])
check("precision", [round(100.0 * s_per_pred[p][1] / s_per_pred[p][0], 1)
                    for p in ("P", "N", "G")], DOM["classbars"]["precision"])

print()
if errors:
    print("MISMATCHES:")
    for e in errors:
        print("  " + e)
else:
    print(f"ALL {14} CHECKS PASS — page numbers == saved-file numbers")
