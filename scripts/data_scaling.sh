#!/usr/bin/env bash
# Does more data help? Train on 1x / 2x / 4x / 8x the episodes of contact_v1 (extra collections with other
# seeds: data/contact_v1_s2 ... s8, same setup) with the default (regularised) settings, all on idle GPUs.
# Each dataset keeps its own split, so contact_v1's test episodes are the same in every run: compare the
# "by dataset" contact_v1 rows of each summary.md, not only the pooled test numbers.
#
#   bash scripts/data_scaling.sh          # -> outputs/scaling/x{1,2,4,8}/
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniconda3/envs/contactsim/bin/python}
OUT=${OUT:-outputs/scaling}
GPUS=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
       awk -F', ' '$2 <= 1000 && $3 <= 10 {printf "%s%s", sep, $1; sep=","}')
echo "idle GPUs: $GPUS"
IFS=, read -r -a G <<< "$GPUS"
rot() { local k=$(( $1 % ${#G[@]} )); (IFS=,; echo "${G[*]:k},${G[*]:0:k}" | sed 's/,$//'); }

declare -A DATA=(
  [x1]="data/contact_v1"
  [x2]="data/contact_v1 data/contact_v1_s2"
  [x4]="data/contact_v1 data/contact_v1_s2 data/contact_v1_s3 data/contact_v1_s4"
  [x8]="data/contact_v1 $(echo data/contact_v1_s{2..8})"
)
declare -A MODELS=([x1]="A B C D" [x2]="B D" [x4]="B D" [x8]="A B C D")

mkdir -p "$OUT"
run() {  # name, rotation
  if [[ -f "$OUT/$1/summary.md" ]]; then echo "skip $1 (done)"; return; fi
  # shellcheck disable=SC2086
  "$PY" scripts/train_baselines.py ${DATA[$1]} --models ${MODELS[$1]} --steps 80000 --eval_every 1000 \
      --gpus "$(rot "$2")" --jobs_per_gpu 2 --out "$OUT/$1" > "$OUT/$1.log" 2>&1 \
    && echo "done $1" || echo "FAILED $1 (see $OUT/$1.log)"
}
# x1, x2, x4 together; x8 alone afterwards: each of its workers holds ~13 GB on its GPU (8 GB of inputs), and
# sharing GPUs with the other runs ran out of memory
run x1 0 & sleep 30; run x2 3 & sleep 30; run x4 6 & wait
run x8 0
