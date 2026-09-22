#!/usr/bin/env python
"""Fetch decision datasets into DecisionItem JSONL.

Uses the HuggingFace datasets-server REST API and only the standard library, so
it runs on the login node's old python (3.6) and does not need `datasets`.

Sources
-------
wanli      3-way NLI (entailment / neutral / contradiction).  Only `train` and
           `test` exist, so a validation slice is carved off the tail of `train`.
banking77  customer-support intent classification.  We keep a fixed set of the
           `--n-classes` most frequent intents and present them as a 26-way (or
           smaller) choice question -- 26 is the ceiling because the logit
           baseline needs single-token labels `A`..`Z`.

    python scripts/00_fetch_data.py --source wanli
    python scripts/00_fetch_data.py --source banking77 --n-classes 26
"""

import argparse
import json
import os
import time

try:
    from urllib.parse import quote
    from urllib.request import Request, urlopen
except ImportError:  # pragma: no cover
    from urllib import quote
    from urllib2 import Request, urlopen

API = "https://datasets-server.huggingface.co/rows"

NLI_OPTIONS = [
    "entailment: the premise entails the hypothesis",
    "neutral: the premise neither entails nor contradicts the hypothesis",
    "contradiction: the premise contradicts the hypothesis",
]
NLI_QUESTION = "What is the relationship between the premise and the hypothesis?"
NLI_GOLD = {"entailment": 0, "neutral": 1, "contradiction": 2}

INTENT_QUESTION = "What is the customer's intent?"

CITIES = ["Lisbon", "Osaka", "Quito", "Riga", "Perth", "Cairo", "Bern", "Lima"]
NAMES = ["Ada", "Bram", "Cleo", "Dario", "Elin", "Faisal", "Greta", "Hugo"]
COLORS = ["teal", "amber", "indigo", "olive"]


def http_json(url, attempts=8):
    last = None
    for i in range(attempts):
        try:
            req = Request(url, headers={"User-Agent": "anylm2jev/0.1"})
            with urlopen(req, timeout=60) as fh:
                return json.loads(fh.read().decode("utf-8"))
        except Exception as exc:  # transient proxy/network wobble (503 tunnels etc.)
            last = exc
            time.sleep(min(2 ** i, 30))
    raise RuntimeError("request failed after %d attempts: %s\n%s" % (attempts, url, last))


def fetch_rows(dataset, split, offset, length, config="default", sleep_s=0.4):
    out = []
    got = 0
    while got < length:
        want = min(100, length - got)
        url = "%s?dataset=%s&config=%s&split=%s&offset=%d&length=%d" % (
            API, quote(dataset, safe=""), config, split, offset + got, want)
        data = http_json(url)
        rows = data.get("rows", [])
        if not rows:
            break
        for r in rows:
            out.append(r["row"])
        got += len(rows)
        if sleep_s:
            time.sleep(sleep_s)
    return out


def humanize(name):
    return str(name).replace("_", " ").strip()


def write(path, records):
    with open(path, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("%s: %d" % (path, len(records)))


# --------------------------------------------------------------------------- WANLI


def nli_record(idx, row, source):
    label = NLI_GOLD.get(str(row.get("gold", "")).strip().lower())
    if label is None:
        return None
    return {
        "item_id": "%s-%d" % (source, idx),
        "state": "Premise: %s\nHypothesis: %s" % (
            str(row["premise"]).strip(), str(row["hypothesis"]).strip()),
        "question": NLI_QUESTION,
        "options": list(NLI_OPTIONS),
        "gold": label,
        "task": "choice",
        "meta": {"source": source, "wanli_id": row.get("id"), "genre": row.get("genre")},
    }


def fetch_wanli(args):
    need_train = args.limit_train + args.limit_val
    raw_train = fetch_rows("alisawuffles/WANLI", "train", 0, need_train)
    raw_test = fetch_rows("alisawuffles/WANLI", "test", 0, args.limit_test)

    train, val, test = [], [], []
    for row in raw_train[: args.limit_train]:
        rec = nli_record(len(train), row, "wanli-train")
        if rec:
            train.append(rec)
    for row in raw_train[args.limit_train:]:
        rec = nli_record(len(val), row, "wanli-val")
        if rec:
            val.append(rec)
    for row in raw_test:
        rec = nli_record(len(test), row, "wanli-test")
        if rec:
            test.append(rec)

    write(os.path.join(args.out_dir, "wanli_train.jsonl"), train)
    write(os.path.join(args.out_dir, "wanli_val.jsonl"), val)
    write(os.path.join(args.out_dir, "wanli_test.jsonl"), test)


# ----------------------------------------------------------------------- BANKING77


def intent_record(idx, row, options, index, source):
    label_text = str(row.get("label_text", "")).strip()
    if label_text not in index:
        return None
    return {
        "item_id": "%s-%d" % (source, idx),
        "state": "Customer message: %s" % str(row["text"]).strip(),
        "question": INTENT_QUESTION,
        "options": list(options),
        "gold": index[label_text],
        "task": "choice",
        "meta": {"source": source, "label_text": label_text},
    }


def fetch_banking77(args):
    if args.n_classes > 77:
        raise SystemExit("banking77 has only 77 intents")
    if args.n_classes > 26:
        print("NOTE: >26 options cannot use the choice format (single-token A..Z "
              "labels); use the noul formulation (scripts/11_induce_noul.py)")

    prefix = args.out_prefix or "banking77"
    pool_train = fetch_rows("mteb/banking77", "train", 0, args.pool_train or 10 ** 9, sleep_s=args.sleep)
    pool_test = fetch_rows("mteb/banking77", "test", 0, args.pool_test or 10 ** 9, sleep_s=args.sleep)
    print("fetched pool: train=%d test=%d" % (len(pool_train), len(pool_test)))

    counts = {}
    for row in pool_train:
        name = str(row.get("label_text", "")).strip()
        counts[name] = counts.get(name, 0) + 1
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[: args.n_classes]
    selected = sorted(name for name, _ in top)  # neutral canonical order
    options = [humanize(name) for name in selected]
    index = {name: i for i, name in enumerate(selected)}
    print("intents (%d): %s" % (len(selected), ", ".join(selected)))

    train_pool, val_pool, test = [], [], []
    for row in pool_train:
        rec = intent_record(0, row, options, index, "banking77")
        if rec:
            train_pool.append(rec)
    for row in pool_test:
        rec = intent_record(0, row, options, index, "banking77-test")
        if rec:
            test.append(rec)

    need = args.limit_train + args.limit_val
    import random as _random
    rng = _random.Random(args.seed)
    rng.shuffle(train_pool)
    rng.shuffle(test)
    pool = train_pool[:need]
    if len(pool) < need:
        print("WARNING: only %d in-class train rows (wanted %d); raise --pool-train"
              % (len(pool), need))
    train = pool[: args.limit_train]
    val = pool[args.limit_train:]
    for i, rec in enumerate(train):
        rec["item_id"] = "banking77-train-%d" % i
        rec["meta"]["source"] = "banking77-train"
    for i, rec in enumerate(val):
        rec["item_id"] = "banking77-val-%d" % i
        rec["meta"]["source"] = "banking77-val"
    for i, rec in enumerate(test):
        rec["item_id"] = "%s-test-%d" % (prefix, i)

    write(os.path.join(args.out_dir, "%s_train.jsonl" % prefix), train)
    write(os.path.join(args.out_dir, "%s_val.jsonl" % prefix), val)
    write(os.path.join(args.out_dir, "%s_test.jsonl" % prefix), test[: args.limit_test])


# ----------------------------------------------------------------------- SYNTHETIC


def synthetic(n, seed):
    import random

    rng = random.Random(seed)
    out = []
    for i in range(n):
        name = rng.choice(NAMES)
        city = rng.choice(CITIES)
        colour = rng.choice(COLORS)
        distractors = rng.sample([c for c in CITIES if c != city], 2)
        options = [city] + distractors
        rng.shuffle(options)
        state = ("%s moved to %s last spring. %s likes the %s trams there and writes "
                 "about them every week." % (name, city, name, colour))
        out.append({
            "item_id": "synth-%d" % i,
            "state": state,
            "question": "Which city does %s live in?" % name,
            "options": options,
            "gold": options.index(city),
            "task": "choice",
            "meta": {"source": "synthetic"},
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["wanli", "banking77", "synthetic"], default="wanli")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--limit-train", type=int, default=3000)
    ap.add_argument("--limit-val", type=int, default=500)
    ap.add_argument("--limit-test", type=int, default=500)
    ap.add_argument("--n-classes", type=int, default=26, help="banking77 fixed option set size (<=77)")
    ap.add_argument("--out-prefix", default="", help="output file prefix (default banking77)")
    ap.add_argument("--pool-train", type=int, default=0, help="banking77 prefix rows to fetch (0 = all; data is label-sorted so a prefix covers the first classes)")
    ap.add_argument("--pool-test", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.4, help="seconds between API requests")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    if args.source == "synthetic":
        write(os.path.join(args.out_dir, "synthetic_train.jsonl"), synthetic(args.limit_train, args.seed))
        write(os.path.join(args.out_dir, "synthetic_val.jsonl"), synthetic(args.limit_val, args.seed + 1))
        write(os.path.join(args.out_dir, "synthetic_test.jsonl"), synthetic(args.limit_test, args.seed + 2))
    elif args.source == "banking77":
        fetch_banking77(args)
    else:
        fetch_wanli(args)


if __name__ == "__main__":
    main()
