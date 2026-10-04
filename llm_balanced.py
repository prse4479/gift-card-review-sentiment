#!/usr/bin/env python3
"""Run the LLM emotion + sentiment pass over the balanced 3-class sample."""
import json
import gzip
import random
import re
import time
import urllib.request

BASE = "http://dobolyi.com:9000/v1/chat/completions"
KEY = __import__("os").environ["HERMES_CUSTOM_DOBOLYI_COM_9000_API_KEY"]
MODEL = "DeepSeek-V4-Flash-0731"
SRC = "/Users/preethiselvaganapathy/Downloads/AI & LLM/sample_balanced.jsonl.gz"
OUT = "/Users/preethiselvaganapathy/Downloads/AI & LLM/sample_balanced_emotions.jsonl.gz"
EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy",
            "sadness", "surprise", "trust"]
BATCH = 10
MAX_TOKENS = 6000

SYSTEM = (
    "You are a review analyst. For each review judge the sentiment "
    "(positive|negative|neutral) and pick exactly ONE primary emotion from: "
    + ", ".join(EMOTIONS) + ".\n"
    'Reply ONLY with a JSON array in the same order: [{"id": <n>, "sentiment": '
    '"positive"|"negative"|"neutral", "primary_emotion": "<one of the 8>"}, ...].'
)


def call_llm(messages, max_tokens=MAX_TOKENS):
    body = json.dumps({"model": MODEL, "messages": messages,
                       "temperature": 0.1, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(BASE, data=body, method="POST", headers={
        "Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=150) as resp:
        msg = json.loads(resp.read().decode())["choices"][0]["message"]
    return msg.get("content") or msg.get("reasoning_content") or ""


def extract_json_array(text):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.S)
    s, e = text.find("["), text.rfind("]")
    if s == -1 or e <= s:
        return None
    try:
        return json.loads(text[s:e + 1])
    except json.JSONDecodeError:
        return None


def main():
    with gzip.open(SRC, "rt", encoding="utf-8") as f:
        sample = [json.loads(l) for l in f]

    results = {}
    for start in range(0, len(sample), BATCH):
        batch = sample[start:start + BATCH]
        got = None
        for attempt in range(3):
            try:
                items = [{"id": r["idx"], "title": r["title"],
                          "text": r["text"][:300]} for r in batch]
                content = call_llm([{"role": "user",
                                     "content": SYSTEM + "\n\nReviews:\n" +
                                     json.dumps(items, ensure_ascii=False)}])
            except Exception as e:
                print(f"  batch {start} err: {e}")
                time.sleep(3)
                continue
            arr = extract_json_array(content)
            if isinstance(arr, list) and len(arr) == len(batch):
                got = arr
                break
        if got is None:
            raise RuntimeError(f"batch {start} unparseable after 3 attempts")
        for obj in got:
            if isinstance(obj, dict) and "id" in obj:
                results[obj["id"]] = obj
        print(f"  done {min(start + BATCH, len(sample))}/{len(sample)}")
        time.sleep(0.5)

    with gzip.open(OUT, "wt", encoding="utf-8") as f:
        for r in sample:
            llm = results.get(r["idx"], {})
            r["llm_sentiment"] = llm.get("sentiment")
            r["llm_emotion"] = llm.get("primary_emotion")
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("completed", len(results), "->", OUT)


if __name__ == "__main__":
    main()
