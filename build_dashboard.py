#!/usr/bin/env python3
"""
Build the results dashboard — Step 6 (3-class key + balanced sample).

Answer key: 4-5★ -> positive, 3★ -> neutral, 1-2★ -> negative.
Evaluation sample: 50 rows per class, fixed seed 2026, from the whole file.
Output: single self-contained HTML file.

Run:  python3 build_dashboard.py
"""
import json
import gzip

SAMPLE = "/Users/preethiselvaganapathy/Downloads/AI & LLM/sample_balanced.jsonl.gz"
FULL = "/Users/preethiselvaganapathy/Downloads/AI & LLM/Gift_Cards_classified_emotions.jsonl.gz"
OUT = "/Users/preethiselvaganapathy/Downloads/AI & LLM/gift_cards_dashboard.html"

CLASSES = ["positive", "neutral", "negative"]


def key_class(rating):
    if rating >= 4:
        return "positive"
    if rating == 3:
        return "neutral"
    return "negative"


def cluster_of(pred, truth):
    if pred == truth:
        return "correct"
    if pred == "neutral":
        return "abstained"
    if pred == "positive":
        return "over_positive"
    return "over_negative"


def pct(a, b):
    return round(100.0 * a / b, 1) if b else 0.0


def main():
    with gzip.open(SAMPLE, "rt", encoding="utf-8") as f:
        sample = [json.loads(l) for l in f]
    with gzip.open(FULL, "rt", encoding="utf-8") as f:
        full_rows = [json.loads(l) for l in f]
    sn, fn = len(sample), len(full_rows)

    # ---- full-dataset 3x3 confusion + per-class metrics --------------------
    conf = {p: {t: 0 for t in CLASSES} for p in CLASSES}
    for r in full_rows:
        conf[r["sentiment"]][key_class(float(r["rating"]))] += 1
    truth_ct = {t: sum(conf[p][t] for p in CLASSES) for t in CLASSES}
    pred_ct = {p: sum(conf[p][t] for t in CLASSES) for p in CLASSES}
    full = {
        "total": fn,
        "label_dist": {p: conf[p][p] for p in CLASSES},
        "confusion": conf,
        "truth": truth_ct,
        "precision": {p: pct(conf[p][p], pred_ct[p]) for p in CLASSES},
        "recall": {t: pct(conf[t][t], truth_ct[t]) for t in CLASSES},
        "agree": pct(sum(conf[p][p] for p in CLASSES), fn),
        "cluster_counts": {
            "abstained": conf["neutral"]["positive"] + conf["neutral"]["negative"],
            "over_positive": conf["positive"]["neutral"] + conf["positive"]["negative"],
            "over_negative": conf["negative"]["positive"] + conf["negative"]["neutral"],
        },
    }

    # ---- balanced-sample stats ---------------------------------------------
    agree = 0
    per_pred = {p: [0, 0] for p in CLASSES}
    per_truth = {t: [0, 0] for t in CLASSES}
    clusters = {"abstained": 0, "over_positive": 0, "over_negative": 0}
    for r in sample:
        ok = r["our_sentiment"] == r["key"]
        agree += ok
        per_pred[r["our_sentiment"]][0] += 1
        per_pred[r["our_sentiment"]][1] += ok
        per_truth[r["key"]][0] += 1
        per_truth[r["key"]][1] += ok
        c = cluster_of(r["our_sentiment"], r["key"])
        if c != "correct":
            clusters[c] += 1
        r["cluster"] = c

    sample_stats = {
        "n": sn,
        "agree": agree,
        "clusters": clusters,
        "per_pred": per_pred,
        "per_truth": per_truth,
        "all": [{
            "rating": r["rating"],
            "key": r["key"],
            "pred": r["our_sentiment"],
            "score": r["our_score"],
            "title": r["title"],
            "text": r["text"],
            "cluster": r["cluster"],
        } for r in sample],
    }

    # ---- star-rating distribution (descriptive) ---------------------------
    star_dist = {s: 0 for s in range(1, 6)}
    for r in full_rows:
        star_dist[int(round(float(r["rating"])))] += 1
    full["star_dist"] = [{
        "star": s,
        "count": star_dist[s],
        "pct": pct(star_dist[s], fn),
        "cls": key_class(s),  # color hint: 4-5★ pos, 3★ neu, 1-2★ neg
    } for s in range(1, 6)]

    # ---- where each balanced-sample class went (descriptive) -------------
    by_key = {t: {p: 0 for p in CLASSES} for t in CLASSES}
    for r in sample:
        by_key[r["key"]][r["our_sentiment"]] += 1
    sample_stats["by_key"] = by_key

    # ---- balanced-sample confusion (rows = predicted, cols = answer key) --
    # identical to the matrix in README section 3
    sample_conf = {p: {t: 0 for t in CLASSES} for p in CLASSES}
    for r in sample:
        sample_conf[r["our_sentiment"]][r["key"]] += 1
    sample_stats["confusion"] = sample_conf

    # ---- deterministic examples per error cluster (full dataset) -----------
    import random
    rng = random.Random(123)
    examples = {}
    for cl, (pred, truth_set) in {
        "abstained": ("neutral", ("positive", "negative")),
        "over_positive": ("positive", ("neutral", "negative")),
        "over_negative": ("negative", ("positive", "neutral")),
    }.items():
        pool = [r for r in full_rows
                if r["sentiment"] == pred and key_class(float(r["rating"])) in truth_set]
        rng.shuffle(pool)
        examples[cl] = [{
            "rating": float(r["rating"]),
            "key": key_class(float(r["rating"])),
            "pred": r["sentiment"],
            "score": r["sentiment_score"],
            "title": (r.get("title") or "")[:70],
            "text": (r.get("text") or "")[:150],
        } for r in pool[:3]]

    data = {
        "meta": {
            "dataset": "Amazon Gift Card reviews",
            "generated": "2026-10-03",
            "answer_key": "4-5★ → positive · 3★ → neutral · 1-2★ → negative",
            "sampling": "balanced: 50 per class, drawn from the whole file",
            "model": "phrase pre-check + VADER (text only; rating never read)",
        },
        "full": full,
        "sample": sample_stats,
        "examples": examples,
    }

    blob = json.dumps(data, ensure_ascii=False, indent=1)
    blob = blob.replace("</", "<\\/")
    html = TEMPLATE.replace("__DATA__", blob)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print("written:", OUT, "({} bytes)".format(len(html)))


TEMPLATE = r"""<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gift Card Review Classifier — Results</title>
<!-- Recolor the whole dashboard by editing the tokens in :root below. -->
<style>
  :root {
    --bg:        #0f1218;   /* page background            */
    --panel:     #171b23;   /* card / panel background    */
    --panel-2:   #1d222c;   /* nested / hover background  */
    --hairline:  #272e3a;   /* borders & dividers         */
    --ink:       #e9ecf2;   /* primary text               */
    --ink-2:     #a7b0c0;   /* secondary text             */
    --ink-3:     #6b7484;   /* tertiary / captions        */
    --accent:    #d9a441;   /* brand accent (gold)         */
    --pos:       #58b383;   /* positive semantic           */
    --neg:       #e27364;   /* negative semantic           */
    --neu:       #7e8ba1;   /* neutral semantic            */
    --radius:    12px;
    --font: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
            "Helvetica Neue", Arial, sans-serif;
    --mono: ui-monospace, "SF Mono", SFMono-Regular, Menlo, Consolas, monospace;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html { background: var(--bg); }
  body {
    font-family: var(--font); font-size: 15px; line-height: 1.55; color: var(--ink);
    background: radial-gradient(1100px 400px at 85% -10%, rgba(217,164,65,.06), transparent 60%), var(--bg);
    -webkit-font-smoothing: antialiased; padding: 48px 28px 72px;
  }
  .wrap { max-width: 1120px; margin: 0 auto; }
  header { margin-bottom: 40px; }
  .kicker { font-size: 11px; letter-spacing: .22em; text-transform: uppercase;
            color: var(--accent); font-weight: 600; margin-bottom: 10px; }
  h1 { font-size: 30px; font-weight: 650; letter-spacing: -.02em; }
  header p.sub { color: var(--ink-2); margin-top: 8px; max-width: 62ch; }
  .meta-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 18px; }
  .chip { font-size: 12px; color: var(--ink-2); background: var(--panel);
          border: 1px solid var(--hairline); border-radius: 999px; padding: 4px 12px;
          font-variant-numeric: tabular-nums; }
  .chip b { color: var(--ink); font-weight: 600; }
  section { margin-top: 52px; }
  .sec-head { margin-bottom: 18px; }
  .sec-head h2 { font-size: 17px; font-weight: 650; letter-spacing: -.01em; }
  .sec-head p { color: var(--ink-2); font-size: 13.5px; margin-top: 4px; max-width: 70ch; }
  .metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }
  @media (max-width: 900px) { .metrics { grid-template-columns: repeat(2, 1fr); } }
  .metric { background: var(--panel); border: 1px solid var(--hairline);
            border-radius: var(--radius); padding: 18px 20px 16px; }
  .metric .label { font-size: 11px; text-transform: uppercase; letter-spacing: .14em;
                   color: var(--ink-3); font-weight: 600; }
  .metric .value { font-size: 34px; font-weight: 700; letter-spacing: -.02em;
                   font-variant-numeric: tabular-nums; margin-top: 6px; line-height: 1.1; }
  .metric .value small { font-size: 17px; font-weight: 600; color: var(--ink-3); }
  .metric .note { font-size: 12px; color: var(--ink-2); margin-top: 6px; }
  .accent-rule { height: 2px; width: 26px; border-radius: 2px; margin-top: 10px; }
  .split { display: grid; grid-template-columns: 300px 1fr; gap: 26px; align-items: center; }
  @media (max-width: 860px) { .split { grid-template-columns: 1fr; } }
  .donut-wrap { display: flex; align-items: center; gap: 22px; }
  .donut-legend { display: grid; gap: 10px; }
  .legend-row { display: flex; align-items: baseline; gap: 10px; font-size: 13.5px; }
  .swatch { width: 10px; height: 10px; border-radius: 3px; flex: none; align-self: center; }
  .legend-row .l { color: var(--ink-2); }
  .legend-row .n { font-variant-numeric: tabular-nums; font-weight: 650; margin-left: auto; }
  .clusters { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
  @media (max-width: 900px) { .clusters { grid-template-columns: 1fr; } }
  .cluster { background: var(--panel); border: 1px solid var(--hairline);
             border-radius: var(--radius); padding: 16px 18px; }
  .cluster .big { font-size: 26px; font-weight: 700; font-variant-numeric: tabular-nums; }
  .cluster h3 { font-size: 13.5px; font-weight: 650; margin: 2px 0 6px; }
  .cluster p { font-size: 12.5px; color: var(--ink-2); }
  .cm { display: grid; grid-template-columns: 150px 1fr 1fr 1fr; gap: 8px; max-width: 760px; }
  .cm .cell { border: 1px solid var(--hairline); border-radius: 10px;
              padding: 14px 16px; background: var(--panel); }
  .cm .cell .n { font-size: 24px; font-weight: 700; font-variant-numeric: tabular-nums; }
  .cm .cell .p { font-size: 12px; color: var(--ink-2); margin-top: 2px; }
  .cm .head { font-size: 11px; text-transform: uppercase; letter-spacing: .14em;
              color: var(--ink-3); font-weight: 600; border: none; background: none; padding: 4px 16px; }
  .cm .rowlab { font-size: 12.5px; color: var(--ink-2); display: flex; align-items: center; }
  .cm .rowlab b { color: var(--ink); }
  .cm .good { border-color: rgba(88,179,131,.35); }
  .cm .bad  { border-color: rgba(226,115,100,.25); }
  .classbars { display: grid; grid-template-columns: 1fr 1fr; gap: 26px; margin-top: 30px; max-width: 700px; }
  @media (max-width: 760px) { .classbars { grid-template-columns: 1fr; } }
  .bar-row { margin-bottom: 14px; }
  .bar-row .bt { display: flex; justify-content: space-between; font-size: 12.5px;
                 color: var(--ink-2); margin-bottom: 5px; }
  .bar-row .bt b { color: var(--ink); font-variant-numeric: tabular-nums; }
  .track { height: 8px; background: var(--panel-2); border-radius: 99px; overflow: hidden; }
  .fill { height: 100%; border-radius: 99px; }

  /* ---- descriptive: star distribution & per-class splits ---- */
  .starrows { display: grid; gap: 10px; max-width: 640px; }
  .star-row .srt { display: flex; justify-content: space-between; font-size: 13px;
                   color: var(--ink-2); margin-bottom: 5px; }
  .star-row .srt b { color: var(--ink); font-variant-numeric: tabular-nums; }
  .star-row .track { height: 14px; }
  .split-cards { display: grid; gap: 14px; }
  .split-card { background: var(--panel); border: 1px solid var(--hairline);
                border-radius: var(--radius); padding: 16px 18px; }
  .split-card .sc-head { display: flex; align-items: baseline; justify-content: space-between;
                         gap: 12px; margin-bottom: 10px; }
  .split-card .sc-head h3 { font-size: 14px; font-weight: 650; }
  .split-card .sc-right { font-size: 13px; color: var(--ink-2); white-space: nowrap;
                          font-variant-numeric: tabular-nums; }
  .split-card .sc-right b { color: var(--ink); font-weight: 700; }
  .splitbar { display: flex; height: 26px; border-radius: 8px; overflow: hidden;
              border: 1px solid var(--hairline); }
  .splitbar .seg { height: 100%; min-width: 0; display: flex; align-items: center;
                   justify-content: center; font-size: 11px; font-weight: 700;
                   color: #0b0e13; }
  .seg-legend { display: flex; gap: 14px; flex-wrap: wrap; margin-top: 8px; font-size: 12px;
                color: var(--ink-2); }
  .seg-legend .sl { display: inline-flex; align-items: center; gap: 6px; }
  .swatch-s { width: 10px; height: 10px; border-radius: 3px; display: inline-block; }
  .verdict { margin-top: 8px; font-size: 12.5px; color: var(--ink-2); }
  .verdict b { font-weight: 650; }
  .examples { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
  @media (max-width: 900px) { .examples { grid-template-columns: 1fr; } }
  .review { background: var(--panel); border: 1px solid var(--hairline);
            border-radius: var(--radius); padding: 16px 18px;
            display: flex; flex-direction: column; gap: 8px; }
  .review .rtitle { font-weight: 650; font-size: 14px; }
  .review .rtext { font-size: 13px; color: var(--ink-2); }
  .review .rmeta { display: flex; gap: 8px; flex-wrap: wrap; margin-top: auto; }
  .tag { font-size: 11px; padding: 2px 9px; border-radius: 999px; border: 1px solid var(--hairline);
         color: var(--ink-2); font-variant-numeric: tabular-nums; }
  .tag.pos { color: var(--pos); border-color: rgba(88,179,131,.4); background: rgba(88,179,131,.08); }
  .tag.neg { color: var(--neg); border-color: rgba(226,115,100,.4); background: rgba(226,115,100,.08); }
  .tag.neu { color: var(--neu); border-color: rgba(126,139,161,.35); background: rgba(126,139,161,.08); }
  .toolbar { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; margin-bottom: 14px; }
  .searchbox { position: relative; flex: 1; min-width: 220px; max-width: 360px; }
  .searchbox input { width: 100%; font: inherit; font-size: 13px; color: var(--ink);
    background: var(--panel); border: 1px solid var(--hairline); border-radius: 999px;
    padding: 8px 16px 8px 36px; outline: none; transition: border-color .15s ease; }
  .searchbox input::placeholder { color: var(--ink-3); }
  .searchbox input:focus { border-color: var(--accent); }
  .searchbox .mag { position: absolute; left: 13px; top: 50%; transform: translateY(-50%);
    color: var(--ink-3); font-size: 14px; line-height: 1; }
  .showcount { font-size: 12.5px; color: var(--ink-2); margin-left: auto;
               font-variant-numeric: tabular-nums; white-space: nowrap; }
  .showcount b { color: var(--ink); font-weight: 650; }
  .filters { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 14px; }
  .fbtn { font: inherit; font-size: 12.5px; cursor: pointer; background: var(--panel);
          color: var(--ink-2); border: 1px solid var(--hairline); border-radius: 999px;
          padding: 5px 14px; transition: all .15s ease; }
  .fbtn .fcnt { display: inline-block; min-width: 18px; text-align: center; margin-left: 6px;
    background: rgba(126,139,161,.15); color: var(--ink-2); font-size: 11px; font-weight: 650;
    border-radius: 99px; padding: 1px 5px; font-variant-numeric: tabular-nums; }
  .fbtn:hover { border-color: var(--ink-3); }
  .fbtn:disabled { opacity: .45; cursor: default; }
  .fbtn:disabled:hover { border-color: var(--hairline); }
  .fbtn.active { background: var(--accent); color: #14100a; border-color: var(--accent); font-weight: 650; }
  .fbtn.active .fcnt { background: rgba(20,16,10,.18); color: #14100a; }
  .pager { display: flex; align-items: center; justify-content: flex-end; gap: 10px;
           margin-top: 14px; }
  .pageinfo { font-size: 12.5px; color: var(--ink-2); font-variant-numeric: tabular-nums; }
  .pageinfo b { color: var(--ink); font-weight: 700; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { text-align: left; font-size: 11px; text-transform: uppercase; letter-spacing: .12em;
       color: var(--ink-3); font-weight: 600; padding: 8px 12px; border-bottom: 1px solid var(--hairline); }
  td { padding: 12px; border-bottom: 1px solid var(--hairline); vertical-align: top; }
  tbody tr { transition: background .12s ease; }
  tbody tr:hover { background: var(--panel-2); }
  td .tt { font-weight: 600; }
  td .tx { color: var(--ink-2); margin-top: 2px; }
  .num { font-variant-numeric: tabular-nums; white-space: nowrap; }
  .dot { display: inline-block; width: 8px; height: 8px; border-radius: 99px; margin-right: 8px; }
  .status { display: inline-flex; align-items: center; gap: 7px; font-weight: 650;
            font-variant-numeric: tabular-nums; }
  .status.ok   { color: var(--pos); }
  .status.bad  { color: var(--neg); }
  .empty { color: var(--ink-3); padding: 22px; text-align: center; font-size: 13px; }
  footer { margin-top: 64px; padding-top: 20px; border-top: 1px solid var(--hairline);
           font-size: 12px; color: var(--ink-3); max-width: 90ch; }
  footer code { font-family: var(--mono); font-size: 11.5px; color: var(--ink-2); }
</style>
</head>
<body>
<div class="wrap">

  <header>
    <h1>Gift Card Review Classifier — Results</h1>
    <p class="sub">A text-only sentiment classifier judged against a three-class
      rating-derived answer key, on a balanced sample of Amazon gift card reviews.</p>
    <div class="meta-row" id="meta"></div>
  </header>

  <section>
    <div class="sec-head">
      <h2>Headline</h2>
      <p>Scored on a balanced 150-review sample (≈50 per class, drawn at fixed random from the
         whole file). The model is text-only; the star rating is used solely to build the answer key afterwards.</p>
    </div>
    <div class="metrics" id="metrics"></div>
  </section>

  <section>
    <div class="sec-head">
      <h2>The data — star-rating distribution</h2>
      <p>All 152,410 reviews, by the stars the reviewer actually gave. Bars are tinted by the
        answer-key class they belong to (4-5★ green, 3★ slate, 1-2★ red). This imbalance is
        exactly why the evaluation uses a balanced sample — otherwise the 3★ and 1-2★ tiers
        would be invisible.</p>
    </div>
    <div id="starDist"></div>
  </section>

  <section>
    <div class="sec-head">
      <h2>Where each class's reviews went</h2>
      <p>The balanced sample (50 reviews per class): for each answer-key class, which label the
        model assigned to those 50 reviews. The green segment is the part answered right; the
        other segments are the failures — visible without any drilling.</p>
    </div>
    <div id="byClass"></div>
  </section>

  <section>
    <div class="sec-head">
      <h2>Right vs wrong</h2>
      <p>Correct calls vs mistakes on the balanced sample, split into three error clusters.</p>
    </div>
    <div class="split">
      <div class="donut-wrap">
        <svg id="donut" width="170" height="170" viewBox="0 0 170 170"></svg>
        <div class="donut-legend" id="donutLegend"></div>
      </div>
      <div class="clusters" id="clusters"></div>
    </div>
  </section>

  <section>
    <div class="sec-head">
      <h2>Confusion matrix — balanced sample (150 reviews)</h2>
      <p>Rows are the model's prediction; columns are the answer-key truth
         (4-5★ positive · 3★ neutral · 1-2★ negative) — the same numbers as
         the results table in the README.</p>
    </div>
    <div class="cm" id="cm"></div>
    <div class="classbars" id="classbars"></div>
  </section>

  <section>
    <div class="sec-head">
      <h2>How mistakes cluster — with examples</h2>
      <p>Every error across the full dataset falls into one of three clusters.
         Real reviews from each, verbatim.</p>
    </div>
    <div id="examples"></div>
  </section>

  <section>
    <div class="sec-head">
      <h2>Per-review detail — all 150 sampled calls</h2>
      <p>Every review in the balanced sample, correct and mismatched alike. Filter by
        outcome, type a keyword to search, and watch the live count update.</p>
    </div>
    <div class="toolbar">
      <div class="searchbox">
        <span class="mag">🔍</span>
        <input id="search" type="text" placeholder="Search title or text…" autocomplete="off">
      </div>
      <div class="showcount" id="showCount"></div>
    </div>
    <div class="filters" id="filters"></div>
    <table id="detail">
      <thead>
        <tr><th>Outcome</th><th>Rating</th><th>Answer key</th><th>Model said</th><th>Score</th><th>Review</th></tr>
      </thead>
      <tbody id="detailBody"></tbody>
    </table>
    <div class="pager" id="pager" style="display:none"></div>
    <p class="empty" id="detailEmpty" style="display:none">No reviews match this filter + search combo.</p>
  </section>

  <footer id="foot"></footer>
</div>

<script>
const DATA = __DATA__;
const $ = (id) => document.getElementById(id);
const FMT = new Intl.NumberFormat("en-US");
const CLS_COLOR = { positive: "var(--pos)", neutral: "var(--neu)", negative: "var(--neg)" };

function tag(label, cls) { return `<span class="tag ${cls}">${label}</span>`; }

function renderMeta() {
  const m = DATA.meta;
  $("meta").innerHTML = [
    `<span class="chip"><b>${m.dataset}</b></span>`,
    `<span class="chip">Model: <b>${m.model}</b></span>`,
    `<span class="chip">Answer key: <b>${m.answer_key}</b></span>`,
    `<span class="chip">Sample: <b>${m.sampling}</b></span>`,
    `<span class="chip">Generated <b>${m.generated}</b></span>`,
  ].join("");
}

function renderStarDist() {
  // each bar's width is capped at 100%; a zero count renders a 2px sliver
  // instead of collapsing, so the row never disappears
  $("starDist").innerHTML =
    `<div class="starrows">` + DATA.full.star_dist.map(s =>
      `<div class="star-row">
         <div class="srt"><span>${"★".repeat(s.star)} (${s.star})</span>
           <span>${FMT.format(s.count)} · <b>${s.pct.toFixed(2)}%</b></span></div>
         <div class="track"><div class="fill" style="width:${
           s.pct > 0 ? Math.max(s.pct, 0.05) : 0.05}%;background:${CLS_COLOR[s.cls]}"></div></div>
       </div>`).join("") + `</div>`;
}

function renderByClass() {
  const by = DATA.sample.by_key;
  const order = ["positive", "neutral", "negative"];
  const segLabel = { positive: "model: positive", neutral: "model: neutral", negative: "model: negative" };
  $("byClass").innerHTML = order.map((t) => {
    const row = by[t];
    const tot = row.positive + row.neutral + row.negative;
    const right = row[t];
    const segments = order.map(p => {
      const w = (100.0 * row[p]) / tot;
      const show = w >= 12 && row[p] > 0;  // only print the number when it fits
      return `<div class="seg" style="flex-basis:${w}%;background:${CLS_COLOR[p]}"
                  title="${segLabel[p]}: ${row[p]}">${show ? row[p] : ""}</div>`;
    }).join("");
    const verdicts = {
      positive: `The ${row.neutral + row.negative} leak(s) went to the other classes (mostly negative).`,
      neutral: `Read as positive ${row.positive}×, as negative ${row.negative}× — only ${right}/${tot} stayed neutral: the 3★ class collapses into the others.`,
      negative: `Missed as positive ${row.positive}×, abstained (neutral) ${row.neutral}×.`,
    }[t];
    return `<div class="split-card">
      <div class="sc-head">
        <h3 style="color:${CLS_COLOR[t]}">Answer key: ${t} (${t === "positive" ? "4-5★" : t === "neutral" ? "3★" : "1-2★"})</h3>
        <div class="sc-right">answered right <b style="color:${right / tot >= 0.5 ? "var(--pos)" : "var(--neg)"}">${right}/${tot}</b> (${(100 * right / tot).toFixed(0)}%)</div>
      </div>
      <div class="splitbar">${segments}</div>
      <div class="seg-legend">
        ${order.map(p => `<span class="sl"><span class="swatch-s" style="background:${CLS_COLOR[p]}"></span>${segLabel[p]} · ${row[p]}</span>`).join("")}
      </div>
      <div class="verdict">${verdicts}</div>
    </div>`;
  }).join("");
}

function renderMetrics() {
  const s = DATA.sample;
  const agree = 100 * s.agree / s.n;
  const cards = [
    ["Agreement", `${agree.toFixed(1)}%`, `${s.agree}/${s.n} calls match the 3-class key`, "var(--accent)"],
    ["Positive recall", `${(100*s.per_truth.positive[1]/s.per_truth.positive[0]).toFixed(1)}%`, "of 4-5★ reviews, caught by the model", "var(--pos)"],
    ["Negative recall", `${(100*s.per_truth.negative[1]/s.per_truth.negative[0]).toFixed(1)}%`, "of 1-2★ reviews, caught by the model", "var(--neg)"],
    ["Neutral recall", `${(100*s.per_truth.neutral[1]/s.per_truth.neutral[0]).toFixed(1)}%`, "of 3★ reviews, caught by the model", "var(--neu)"],
  ];
  $("metrics").innerHTML = cards.map(([label, value, note, color]) =>
    `<div class="metric">
       <div class="label">${label}</div>
       <div class="value">${value}</div>
       <div class="note">${note}</div>
       <div class="accent-rule" style="background:${color}"></div>
     </div>`).join("");
}

function renderSplit() {
  const s = DATA.sample;
  const C = 2 * Math.PI * 62;
  const segs = [
    { v: s.agree, color: "var(--pos)" },
    { v: s.n - s.agree, color: "var(--neg)" },
  ];
  let off = 0;
  const circles = segs.map(p => {
    const frac = p.v / s.n;
    const c = `<circle cx="85" cy="85" r="62" fill="none" stroke="${p.color}" stroke-width="20"
      stroke-dasharray="${(frac*C).toFixed(1)} ${C.toFixed(1)}" stroke-dashoffset="${(-off).toFixed(1)}"
      transform="rotate(-90 85 85)"/>`;
    off += frac * C;
    return c;
  }).join("");
  $("donut").innerHTML =
    `<circle cx="85" cy="85" r="62" fill="none" stroke="var(--panel-2)" stroke-width="20"/>` + circles +
    `<text x="85" y="81" text-anchor="middle" fill="var(--ink)" font-size="27" font-weight="700">${Math.round(100*s.agree/s.n)}%</text>
     <text x="85" y="100" text-anchor="middle" fill="var(--ink-3)" font-size="10" letter-spacing="1">RIGHT</text>`;
  $("donutLegend").innerHTML = [
    ["Correct", s.agree, "var(--pos)"],
    ["Wrong", s.n - s.agree, "var(--neg)"],
    ["of which: abstained (neutral)", s.clusters.abstained, "var(--neu)"],
    ["of which: over-positive", s.clusters.over_positive, "var(--neg)"],
    ["of which: over-negative", s.clusters.over_negative, "var(--neg)"],
  ].map(([l, v, color]) =>
    `<div class="legend-row"><span class="swatch" style="background:${color}"></span>
     <span class="l">${l}</span><span class="n">${v}</span></div>`).join("");

  $("clusters").innerHTML = [
    ["var(--neu)", "Abstained", s.clusters.abstained,
     "Model said “neutral” (no sentiment words in text) but the key says positive or negative."],
    ["var(--neg)", "Over-positive", s.clusters.over_positive,
     "Model said positive; key says 3★ or 1-2★. Complaints buried under positive vocabulary."],
    ["var(--neg)", "Over-negative", s.clusters.over_negative,
     "Model said negative; key says 3★ or 4-5★. Critical wording from a satisfied reviewer."],
  ].map(([color, title, n, desc]) =>
    `<div class="cluster"><div class="big" style="color:${color}">${n}</div>
     <h3>${title}</h3><p>${desc}</p></div>`).join("");
}

function renderCM() {
  const s = DATA.sample, conf = s.confusion;
  const order = ["positive", "neutral", "negative"];
  let html = `<div class="head"></div>` +
    order.map(t => `<div class="head" style="color:${CLS_COLOR[t]}">Truth: ${t === "positive" ? "positive (4-5★)" : t === "neutral" ? "neutral (3★)" : "negative (1-2★)"}</div>`).join("");
  for (const p of order) {
    html += `<div class="rowlab"><b style="color:${CLS_COLOR[p]}">Model: ${p}</b></div>`;
    for (const t of order) {
      const v = conf[p][t];
      const denom = s.per_truth[t][0];
      const cls = p === t ? "good" : "bad";
      html += `<div class="cell ${cls}">
                 <div class="n" style="color:${CLS_COLOR[p]}">${FMT.format(v)}</div>
                 <div class="p">${(100 * v / denom).toFixed(1)}% of true ${t}</div>
               </div>`;
    }
  }
  $("cm").innerHTML = html;

  const bars = [];
  for (const t of order) {
    const [tot, corr] = s.per_truth[t];
    const rec = tot ? 100 * corr / tot : 0;
    bars.push(`<div class="bar-row"><div class="bt"><span>Recall — true <b style="color:${CLS_COLOR[t]}">${t}</b></span><b>${rec.toFixed(1)}%</b></div>
      <div class="track"><div class="fill" style="width:${rec}%;background:${CLS_COLOR[t]}"></div></div></div>`);
  }
  for (const p of order) {
    const [tot, corr] = s.per_pred[p];
    const prec = tot ? 100 * corr / tot : 0;
    bars.push(`<div class="bar-row"><div class="bt"><span>Precision — model says <b style="color:${CLS_COLOR[p]}">${p}</b></span><b>${prec.toFixed(1)}%</b></div>
      <div class="track"><div class="fill" style="width:${prec}%;background:${CLS_COLOR[p]}"></div></div></div>`);
  }
  $("classbars").innerHTML = `<div>${bars.slice(0,3).join("")}</div><div>${bars.slice(3,6).join("")}</div>`;
}

function reviewCard(r) {
  return `<div class="review">
    <div class="rtitle">${r.title || "(no title)"}</div>
    <div class="rtext">${r.text || "(no text)"}</div>
    <div class="rmeta">
      ${tag(`${r.rating.toFixed(1)}★ · key: ${r.key}`, r.key)}
      ${tag(`model: ${r.pred}`, r.pred)}
      ${tag(`score ${r.score >= 0 ? "+" : ""}${r.score.toFixed(3)}`, "neu")}
    </div></div>`;
}

function renderExamples() {
  const ex = DATA.examples, cc = DATA.full.cluster_counts;
  const sections = [
    ["Abstained — model said neutral, key says pos/neg", "var(--neu)", "abstained", ex.abstained],
    ["Over-positive — model said positive, key says 3★ or lower", "var(--neg)", "over_positive", ex.over_positive],
    ["Over-negative — model said negative, key says 3★ or higher", "var(--neg)", "over_negative", ex.over_negative],
  ];
  $("examples").innerHTML = sections.map(([title, color, key, list]) => `
    <div style="margin-bottom:26px">
      <h3 style="font-size:14px;font-weight:650;color:${color};margin-bottom:10px">
        ${title} <span style="color:var(--ink-3);font-weight:500">· ${FMT.format(cc[key])} across the full dataset</span></h3>
      <div class="examples">${list.map(reviewCard).join("")}</div>
    </div>`).join("");
}

const FILTER_ORDER = [
  ["all", "All reviews"],
  ["correct", "Correct ✓"],
  ["mismatched", "Mismatched ✗"],
  ["abstained", "Abstained (neutral)"],
  ["over_positive", "Over-positive"],
  ["over_negative", "Over-negative"],
];
const PAGE_SIZE = 20;
let state = { filter: "all", query: "", page: 0 };

function rowMatches(r, key) {
  if (key === "all") return true;
  if (key === "correct") return r.cluster === "correct";
  if (key === "mismatched") return r.cluster !== "correct";
  return r.cluster === key;
}
function rowMatchesQuery(r, q) {
  if (!q) return true;
  return ((r.title || "") + " " + (r.text || "")).toLowerCase().includes(q);
}

function renderPager(matchCount, totalPages) {
  const pager = $("pager");
  if (matchCount <= PAGE_SIZE) {
    pager.style.display = "none";
    return;
  }
  pager.style.display = "flex";
  pager.innerHTML =
    `<button class="fbtn pg" data-dir="-1" ${state.page === 0 ? "disabled" : ""}>← Prev</button>
     <span class="pageinfo">Page <b>${state.page + 1}</b> of ${totalPages}</span>
     <button class="fbtn pg" data-dir="1" ${state.page >= totalPages - 1 ? "disabled" : ""}>Next →</button>`;
  pager.querySelectorAll(".pg").forEach(b =>
    b.addEventListener("click", () => {
      if (b.disabled) return;
      state.page += parseInt(b.dataset.dir, 10);
      renderExplorer();
    }));
}

function renderExplorer() {
  const rows = DATA.sample.all;
  const q = state.query.trim().toLowerCase();
  const visible = rows.filter(r => rowMatches(r, state.filter) && rowMatchesQuery(r, q));
  const totalPages = Math.max(1, Math.ceil(visible.length / PAGE_SIZE));
  if (state.page >= totalPages) state.page = totalPages - 1;

  const start = state.page * PAGE_SIZE;
  const pageRows = visible.slice(start, start + PAGE_SIZE);
  const end = start + pageRows.length;

  $("showCount").innerHTML = visible.length
    ? `Showing <b>${start + 1}–${end}</b> of <b>${visible.length}</b>`
    : `Showing <b>0</b> of <b>0</b>`;

  $("detailBody").innerHTML = pageRows.map(row => {
    const ok = row.cluster === "correct";
    return `<tr>
      <td><span class="status ${ok ? "ok" : "bad"}">
            <span class="dot" style="background:${ok ? "var(--pos)" : "var(--neg)"}"></span>
            ${ok ? "✓" : "✗"} ${row.cluster === "correct" ? "match" : row.cluster === "abstained" ? "abstained" : "mismatch"}</span></td>
      <td class="num">${row.rating.toFixed(1)}★</td>
      <td><span class="dot" style="background:${CLS_COLOR[row.key]}"></span>${row.key}</td>
      <td><span class="dot" style="background:${CLS_COLOR[row.pred]}"></span>${row.pred}</td>
      <td class="num">${row.score >= 0 ? "+" : ""}${row.score.toFixed(3)}</td>
      <td><div class="tt">${row.title || "(no title)"}</div><div class="tx">${row.text || ""}</div></td>
    </tr>`;
  }).join("");
  $("detailEmpty").style.display = visible.length ? "none" : "block";
  renderPager(visible.length, totalPages);
}

function renderFilters() {
  const rows = DATA.sample.all;
  const counts = { all: rows.length };
  for (const r of rows) counts[r.cluster] = (counts[r.cluster] || 0) + 1;
  counts.mismatched = counts.abstained + counts.over_positive + counts.over_negative;
  $("filters").innerHTML = FILTER_ORDER.map(([key, label]) =>
    `<button class="fbtn${state.filter === key ? " active" : ""}" data-f="${key}">${label}<span class="fcnt">${counts[key] || 0}</span></button>`).join("");
  document.querySelectorAll(".fbtn").forEach(b =>
    b.addEventListener("click", () => {
      document.querySelectorAll(".fbtn").forEach(x => x.classList.remove("active"));
      b.classList.add("active");
      state.filter = b.dataset.f;
      state.page = 0;
      renderExplorer();
    }));
}
function initExplorer() {
  renderFilters();
  renderExplorer();
  $("search").addEventListener("input", e => {
    state.query = e.target.value;
    state.page = 0;
    renderExplorer();
  });
}

function renderFoot() {
  const f = DATA.full;
  $("foot").innerHTML =
    `Method: the model reads only <code>title</code> + <code>text</code> (phrase pre-check, then VADER;
     compound &lt; 0 → negative, == 0 → neutral, &gt; 0 → positive). The rating is never an input; it is read
     back afterwards strictly to build the 3-class answer key (<code>4-5★ → positive, 3★ → neutral, 1-2★ → negative</code>).
     The headline and detail table use a balanced sample (50 per class, drawn at fixed random
     from the whole file) so the rare classes are not drowned out;
     the confusion matrix above uses that same balanced sample. The full dataset
     (${FMT.format(f.total)} reviews) has a true split of
     ${FMT.format(f.truth.positive)} positive, ${FMT.format(f.truth.neutral)} neutral, ${FMT.format(f.truth.negative)} negative.
     Rebuild with <code>python3 build_dashboard.py</code>; recolor via the CSS tokens in <code>:root</code>.`;
}

renderMeta(); renderMetrics(); renderStarDist(); renderByClass(); renderSplit(); renderCM(); renderExamples(); initExplorer(); renderFoot();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
