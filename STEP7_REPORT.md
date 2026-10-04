# STEP 7 REPORT — Descriptive layer + verification

## What was added to the dashboard

1. **Star-rating distribution** (full dataset, 152,410 reviews):
   ★1 12,326 (8.1%) · ★2 1,873 (1.2%) · ★3 3,271 (2.1%) · ★4 6,692 (4.4%) · ★5 128,248 (84.1%)
   Bars tinted by answer-key class (1-2★ red, 3★ slate, 4-5★ green) so the
   imbalance is visible before any sampling explanation.

2. **"Where each class's reviews went"** — for each answer-key class in the
   balanced sample, a split bar showing how the model distributed its 50
   reviews (green = right, red/slate = failures), with per-segment counts,
   an "answered right N/50" readout, and a plain-language verdict:
   - Positive (4-5★): **49/50 right (98%)** — 1 leak to negative.
   - Neutral (3★): **2/50 right (4%)** — 35 read as positive, 13 as negative.
     Verdict: "the 3★ class collapses into the others." ← the failure is
     visible at a glance, no drilling.
   - Negative (1-2★): **31/50 right (62%)** — 11 missed as positive, 8 abstained.

3. Existing sections kept: headline metrics, right-vs-wrong donut + error
   clusters, full-dataset 3x3 confusion matrix (+precision/recall bars),
   cluster examples, and the explorable 150-row table.

## Verification (in the browser, not by eye)

- Scraped every number off the rendered page via the live DOM, then
  recomputed the same quantities from the saved data files and asserted
  equality: **ALL 14 CHECKS PASS** (headline metrics, donut %, clusters,
  all 9 confusion cells, star counts + percents, all 3 split-bar blocks
  incl. segment counts and right/total, filter counts, table row count,
  precision/recall bars). Script: `verify_dashboard.py`.
  (Two false mismatches during the run were checker bugs — metric ordering
  and rounding precision — fixed in the checker, not the page.)
- Layout check: measured the computed widths of all 20 chart elements
  (.fill bars and .seg split segments): **zero zero-width elements, zero
  collapsed elements**. The 1★→absence hazard (a bar given ~0 width) is
  guarded by a minimum 0.05% width sliver and flex-basis percentage
  segments with `min-width: 0`; counts print only when the segment is
  >= 12% wide, with hover titles carrying the exact number.

## The balanced-sample summary (unchanged by this step)

Agreement 54.7% (82/150) · recall: positive 98%, negative 62%, neutral 4%
· errors: 8 abstained, 46 over-positive, 14 over-negative.

## Bonus result — LLM take on the balanced sample (just completed)

| | lexicon (VADER+phrases) | LLM |
|---|---|---|
| agreement vs 3-class key | 54.7% | **74.0%** (111/150) |
| positive recall | 98% | 94% |
| negative recall | 62% | **98%** |
| neutral (3★) recall | 4% (2/50) | **30%** (15/50) — still drifts: 32→negative, 15→neutral, 3→positive |

So the LLM does *not* collapse 3★ into positive the way the lexicon does,
but it over-rotates 3★ toward negative. File: `sample_balanced_emotions.jsonl.gz`.

## Deliverables (all in Downloads/AI & LLM)

| File | Role |
|---|---|
| `llm_prompt.txt` | the LLM prompt (both sentiment + primary emotion) |
| `classify_gift_cards.py` | scoring script (runs all reviews, text-only) |
| `nrc_score.py` | word-list script (adds NRC emotions, full dataset) |
| `build_balanced_sample.py` | balanced 50/class sample builder (seed 2026) |
| `llm_balanced.py` | LLM batched scoring over the balanced sample |
| `build_dashboard.py` | dashboard generator (single HTML, recolors via :root) |
| `gift_cards_dashboard.html` | the dashboard itself (offline, self-contained) |
| `eval_batch.py` | 3-class evaluator against the answer key |
| `sample_balanced.jsonl.gz` | the fixed balanced sample |
| `sample_balanced_emotions.jsonl.gz` | same sample + LLM + NRC emotions |
| `Gift_Cards_classified_emotions.jsonl.gz` | full dataset + NRC emotions |
| `verify_dashboard.py` | the browser-vs-files number check |
