#!/usr/bin/env bash
# End-to-end pipeline on a single GPU.  No scheduler and no cluster assumptions.
#
#   ./run.sh data            fetch WANLI + Banking77 into data/        (CPU, network)
#   ./run.sh wanli           3-way NLI:  induce -> distill -> eval
#   ./run.sh banking77       26-way intent, debiased head
#   ./run.sh banking77_full  77-way intent, option-conditioned (noul)
#   ./run.sh figures         comparison figures -> docs/figures/
#   ./run.sh all             everything above
#
# Override anything with env vars, e.g.
#   MODEL=Qwen/Qwen2.5-1.5B-Instruct DEVICE=cuda ./run.sh banking77
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=src

PYTHON=${PYTHON:-python}
DEVICE=${DEVICE:-cuda}
MODEL=${MODEL:-Qwen/Qwen3.5-2B-Base}
EPOCHS=${EPOCHS:-40}
BATCH_SIZE=${BATCH_SIZE:-8}

banner() { printf '\n== %s\n' "$*"; }

cmd_data() {
  banner "fetching data"
  "$PYTHON" scripts/00_fetch_data.py --source wanli \
    --limit-train 3000 --limit-val 500 --limit-test 500
  "$PYTHON" scripts/00_fetch_data.py --source banking77 --n-classes 26 \
    --limit-train 2000 --limit-val 400 --limit-test 500 --pool-train 3600 --pool-test 1200
  "$PYTHON" scripts/00_fetch_data.py --source banking77 --n-classes 77 --out-prefix banking77full \
    --limit-train 2000 --limit-val 400 --limit-test 500 --sleep 0.6
}

# args: tag train_jsonl test_jsonl paraphrases max_orders max_length
choice_round() {
  local tag=$1 train=$2 test=$3 para=$4 orders=$5 len=$6
  banner "$tag: induce"
  "$PYTHON" scripts/10_induce.py --model "$MODEL" --device "$DEVICE" --dtype bfloat16 \
    --data "$train" --out "runs/induce/${tag}_train" \
    --paraphrases "$para" --max-orders "$orders" --max-length "$len" --batch-size "$BATCH_SIZE"
  "$PYTHON" scripts/10_induce.py --model "$MODEL" --device "$DEVICE" --dtype bfloat16 \
    --data "$test" --out "runs/induce/${tag}_test" \
    --paraphrases "$para" --max-orders "$orders" --max-length "$len" --batch-size "$BATCH_SIZE"

  banner "$tag: distil a debiased head"
  "$PYTHON" scripts/20_train_head.py --device "$DEVICE" \
    --induce "runs/induce/${tag}_train" --out "runs/head/${tag}_db.pt" \
    --views all --epochs "$EPOCHS" --debias batch --combine logmean

  banner "$tag: eval"
  "$PYTHON" scripts/30_eval.py --device "$DEVICE" \
    --induce "runs/induce/${tag}_train" --head "runs/head/${tag}_db.pt" \
    --test-induce "runs/induce/${tag}_test" --out "runs/eval/${tag}.json"

  "$PYTHON" scripts/45_debias_eval.py --device cpu \
    --induce "runs/induce/${tag}_train" --test-induce "runs/induce/${tag}_test" \
    --head "runs/head/${tag}_db.pt" --priors none,batch --combines mean,logmean \
    --out "runs/eval/fig_${tag}.json"
}

cmd_wanli() { choice_round wanli_qwen35 data/wanli_train.jsonl data/wanli_test.jsonl 4 0 512; }

cmd_banking77() {
  choice_round banking77_qwen35 data/banking77_train.jsonl data/banking77_test.jsonl 2 3 768
  banner "Banking77 26-way: paired significance"
  "$PYTHON" scripts/40_analyze.py --device cpu \
    --induce runs/induce/banking77_qwen35_train \
    --test-induce runs/induce/banking77_qwen35_test \
    --head runs/head/banking77_qwen35_db.pt --out runs/eval/banking77_qwen35_sig.json
}

cmd_banking77_full() {
  local tag=b77full_noul
  banner "$tag: induce (option-conditioned yes/no, any K)"
  "$PYTHON" scripts/11_induce_noul.py --model "$MODEL" --device "$DEVICE" --dtype bfloat16 \
    --data data/banking77full_train.jsonl --out "runs/induce/${tag}_train" \
    --paraphrases 2 --max-length 512 --batch-size 16
  "$PYTHON" scripts/11_induce_noul.py --model "$MODEL" --device "$DEVICE" --dtype bfloat16 \
    --data data/banking77full_test.jsonl --out "runs/induce/${tag}_test" \
    --paraphrases 2 --max-length 512 --batch-size 16

  banner "$tag: distil + eval"
  "$PYTHON" scripts/20_train_head.py --device "$DEVICE" \
    --induce "runs/induce/${tag}_train" --out "runs/head/${tag}.pt" \
    --views all --epochs "$EPOCHS" --debias batch --combine mean
  "$PYTHON" scripts/30_eval.py --device "$DEVICE" \
    --induce "runs/induce/${tag}_train" --head "runs/head/${tag}.pt" \
    --test-induce "runs/induce/${tag}_test" --out "runs/eval/${tag}.json"
}

cmd_figures() {
  banner "figures"
  "$PYTHON" scripts/70_figures.py
}

case "${1:-all}" in
  data)            cmd_data ;;
  wanli)           cmd_wanli ;;
  banking77)       cmd_banking77 ;;
  banking77_full)  cmd_banking77_full ;;
  figures)         cmd_figures ;;
  all)             cmd_data; cmd_wanli; cmd_banking77; cmd_banking77_full; cmd_figures ;;
  *) echo "usage: $0 {data|wanli|banking77|banking77_full|figures|all}" >&2; exit 2 ;;
esac
echo "done"
