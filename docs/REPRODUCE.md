# Reproduce

All numbers in [`RESULTS.md`](RESULTS.md) come from the commands below.  The
pipeline is split into stages so the expensive part (running the LM over views) is
done once and everything after it is cheap and CPU-runnable.  `run.sh` wraps the
common paths:

```bash
./run.sh data | wanli | banking77 | banking77_full | figures | all
```

## 0. Environment

One GPU is enough (~24 GB); the analysis and figure stages run on CPU.

```bash
python -m venv .venv && . .venv/bin/activate      # or: conda env create -f environment.yml
pip install -r requirements.txt
export PYTHONPATH=src
```

`Qwen/Qwen3.5-2B-Base` is a `qwen3_5` hybrid linear-attention checkpoint and needs
`transformers>=5.17`; older versions reject the architecture outright.  Any
Qwen2/3-class instruct model works as a drop-in via `--model`.

`src/jev/` deliberately keeps `torch`/`transformers` out of the pure data/schema
modules, so `python -m unittest discover -s tests` runs with numpy only.

## 1. Data

`scripts/00_fetch_data.py` is stdlib-only and talks to the HuggingFace
datasets-server REST API, so it needs no GPU (network required on first run).

```bash
python scripts/00_fetch_data.py --source wanli \
  --limit-train 3000 --limit-val 500 --limit-test 500

# 26-way intent: the choice format is capped at 26 single-token labels (A..Z)
python scripts/00_fetch_data.py --source banking77 --n-classes 26 \
  --limit-train 2000 --limit-val 400 --limit-test 500 --pool-train 3600 --pool-test 1200

# 77-way intent (noul format; no label ceiling)
python scripts/00_fetch_data.py --source banking77 --n-classes 77 --out-prefix banking77full \
  --limit-train 2000 --limit-val 400 --limit-test 500 --sleep 0.6
```

`mteb/banking77` is sorted by label, so a contiguous prefix only covers the first
few classes.  The 26-way recipe fetches a prefix; the 77-way recipe fetches the
whole split.

## 2. Choice format (single context, label logits)

```bash
# stage 1: run the frozen LM over every (paraphrase, option-order) view
PYTHONPATH=src python scripts/10_induce.py --model Qwen/Qwen3.5-2B-Base \
  --data data/banking77_train.jsonl --out runs/induce/banking77_train \
  --paraphrases 2 --max-orders 3 --max-length 768
PYTHONPATH=src python scripts/10_induce.py --model Qwen/Qwen3.5-2B-Base \
  --data data/banking77_test.jsonl --out runs/induce/banking77_test \
  --paraphrases 2 --max-orders 3 --max-length 768

# stage 2: distil the *debiased* q* into the head (head-only, no LM)
PYTHONPATH=src python scripts/20_train_head.py \
  --induce runs/induce/banking77_train --out runs/head/banking77_db.pt \
  --views all --epochs 40 --debias batch --combine logmean

# stage 3: eval on the held-out split
PYTHONPATH=src python scripts/30_eval.py \
  --induce runs/induce/banking77_train --head runs/head/banking77_db.pt \
  --test-induce runs/induce/banking77_test --out runs/eval/banking77_qwen35.json
```

For the 3-way WANLI run use `--paraphrases 4 --max-orders 0 --max-length 512`
(`0` = identity plus all rotations, cheap and complete for 3 options).

`--max-orders` matters: `0` means identity plus all cyclic rotations, which is fine
for 3 options but would be 26 (or 77) views for a large option set, so the
high-cardinality runs use 3 seeded random permutations instead.  That is not just a
cost trade-off -- measuring both at K=26 showed the all-K schedule is slightly
*worse* (see [`RESULTS.md`](RESULTS.md)).  For a `q*`-only comparison add
`--no-hidden` to skip storing hidden states (~10x smaller on disk;
`45_debias_eval.py` then runs without `--head`):

```bash
PYTHONPATH=src python scripts/10_induce.py --model Qwen/Qwen3.5-2B-Base \
  --data data/banking77_test.jsonl --out runs/induce/banking77_allk_test \
  --paraphrases 2 --max-orders 0 --max-length 768 --no-hidden
```

## 3. Noul format (option-conditioned yes/no, arbitrary K)

```bash
PYTHONPATH=src python scripts/11_induce_noul.py --model Qwen/Qwen3.5-2B-Base \
  --data data/banking77full_train.jsonl --out runs/induce/b77full_noul_train \
  --paraphrases 2 --max-length 512
PYTHONPATH=src python scripts/11_induce_noul.py --model Qwen/Qwen3.5-2B-Base \
  --data data/banking77full_test.jsonl --out runs/induce/b77full_noul_test \
  --paraphrases 2 --max-length 512

PYTHONPATH=src python scripts/20_train_head.py \
  --induce runs/induce/b77full_noul_train --out runs/head/b77full_noul.pt \
  --views all --epochs 40 --debias batch --combine mean

PYTHONPATH=src python scripts/30_eval.py \
  --induce runs/induce/b77full_noul_train --head runs/head/b77full_noul.pt \
  --test-induce runs/induce/b77full_noul_test --out runs/eval/b77full_noul.json
```

`11_induce_noul.py` writes the **same array contract** as the choice pipeline, so
stages 2 and 3 are reused unchanged.

## 4. LoRA (adapt the backbone, not just the readout)

```bash
PYTHONPATH=src python scripts/25_train_lora.py \
  --induce runs/induce/banking77_train --out runs/lora/banking77 \
  --epochs 3 --lora-r 8 --batch-items 4 --max-length 768

PYTHONPATH=src python scripts/35_eval_lora.py \
  --induce runs/induce/banking77_train --test-induce runs/induce/banking77_test \
  --lora-dir runs/lora/banking77 --head runs/lora/banking77/head.pt \
  --out runs/eval/banking77_lora.json
```

Unlike stage 2, this stage runs the LM (so gradients reach the adapters) rather
than reading stored hidden states.  Two pitfalls are already handled in code and
are easy to reintroduce:

- **dtype**: peft builds adapters in fp32 while the checkpoint is bf16; the
  adapters are explicitly aligned, and the head consumes `hidden.float()`.
- **target order**: head scores come out in *display* order, so `q*` must be
  permuted with `view.order` before the KL loss.  The identity view hides this bug
  completely, which is why it is called out here.

## 5. Label-free debiasing

The prior correction needs no labels, and the **batch** estimator needs no forward
pass either -- it is just the mean of the stored per-view distributions.  The
content-free estimator does need one pass per item x view x probe.

```bash
# batch prior: no new compute, just train a debiased target (--debias batch above)
# content-free prior: one pass per item x view x probe
PYTHONPATH=src python scripts/12_induce_prior_cf.py --model Qwen/Qwen3.5-2B-Base \
  --data data/banking77_train.jsonl --out runs/prior/banking77_cf_train \
  --paraphrases 2 --max-orders 3
PYTHONPATH=src python scripts/12_induce_prior_cf.py --model Qwen/Qwen3.5-2B-Base \
  --data data/banking77_test.jsonl --out runs/prior/banking77_cf_test \
  --paraphrases 2 --max-orders 3

# compare none / batch / content_free, mean / logmean, with coverage@5% and CIs
PYTHONPATH=src python scripts/45_debias_eval.py \
  --induce runs/induce/banking77_train --test-induce runs/induce/banking77_test \
  --head runs/head/banking77_db.pt \
  --priors none,batch,content_free --combines mean,logmean \
  --content-free-train runs/prior/banking77_cf_train.npz \
  --content-free-test runs/prior/banking77_cf_test.npz \
  --out runs/eval/banking77_cf.json
```

`45_debias_eval.py` is CPU-only: it reconstructs position space from the stored
`view_order` and never touches the LM.  **The debiasing must go into the training
target** (`20_train_head.py --debias ...`), not onto a trained head; see
[`RESULTS.md`](RESULTS.md) for the 0.762 -> 0.268 collapse that motivates the rule.

## 6. Analysis and figures

```bash
# paired bootstrap CI + exact McNemar on the discordant pairs (CPU)
PYTHONPATH=src python scripts/40_analyze.py \
  --induce runs/induce/banking77_train --test-induce runs/induce/banking77_test \
  --head runs/head/banking77_db.pt --out runs/eval/banking77_qwen35_sig.json

# comparison figures (scikit-learn CalibrationDisplay + matplotlib)
PYTHONPATH=src python scripts/70_figures.py      # writes docs/figures/*.png
```

`70_figures.py` reads `runs/eval/fig_*.json`, which `45_debias_eval.py` writes
(with the reliability and risk-coverage curves included).

## Smoke test

```bash
PYTHONPATH=src python scripts/smoke.py --limit 4    # loads a model, checks the readout
python -m unittest discover -s tests                # numpy-only, no GPU
```

## Reading the artefacts

```text
runs/induce/<tag>.npz         per-view logits, probabilities and hidden states + q*
runs/induce/<tag>.items.jsonl the DecisionItem records, in order
runs/head/<tag>.pt            head weights + the validation temperature + args
runs/eval/<tag>.json          full metric rows, reliability curves, risk-coverage
runs/prior/<tag>.npz          content-free probe distributions
docs/figures/*.png            generated comparison figures
```
