#!/usr/bin/env bash
# Regularisation sweep for the overfitting seen in the first baselines (val AP peaks within ~1 epoch).
# Models B and D (the draft's history question), 3 seeds, long training (20k steps) to see whether curves converge.
# (r9 = r3 with 40k steps was run by hand: --steps 40000 --eval_every 500.)
# Settings are compared on val AP only; test sets are left for the final A-D run.
#
#   bash scripts/sweep_regularization.sh            # -> outputs/sweep/<name>/, then compare_runs.py
#
# Runs PARALLEL configurations at once on every idle GPU (several jobs per GPU; the models are small).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniconda3/envs/contactsim/bin/python}
OUT=${OUT:-outputs/sweep}
PARALLEL=${PARALLEL:-3}
GPUS=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
       awk -F', ' '$2 <= 1000 && $3 <= 10 {printf "%s%s", sep, $1; sep=","}')
echo "idle GPUs: $GPUS"

# every configuration starts from "no regularisation" (later arguments win), so it does not depend on the
# script defaults, which are now the setting this sweep selected
BASE="--dropout 0 --weight_decay 1e-4 --force_noise 0 --force_bias 0 --joint_noise 0 --posture_shift 0"
SENSOR="--force_noise 0.1 --force_bias 0.1 --joint_noise 0.005"
REG="--dropout 0.3 --weight_decay 0.01"
declare -A CONFIGS=(
  [r0_none]=""
  [r1_sensor]="$SENSOR"
  [r2_sensor_shift03]="$SENSOR --posture_shift 0.03"
  [r3_sensor_shift06]="$SENSOR --posture_shift 0.06"
  [r4_dropout_wd]="$REG"
  [r5_all]="$SENSOR --posture_shift 0.03 $REG"
  [r6_all_lstm32]="$SENSOR --posture_shift 0.03 $REG --lstm_hidden 32 --head_hidden 32"
  [r7_sensor_shift10]="$SENSOR --posture_shift 0.1"
  [r8_all_shift06]="$SENSOR --posture_shift 0.06 $REG"
  [d1_none_data25]="--train_frac 0.25"
  [d2_none_data50]="--train_frac 0.5"
  [d3_all_data25]="$SENSOR --posture_shift 0.03 $REG --train_frac 0.25"
  [d4_all_data50]="$SENSOR --posture_shift 0.03 $REG --train_frac 0.5"
)

mkdir -p "$OUT"
IFS=, read -r -a GPU_LIST <<< "$GPUS"
run() {  # name, index (rotates the GPU list so parallel configurations spread over all GPUs)
  local name=$1 k=$(( $2 % ${#GPU_LIST[@]} ))
  local gpus; gpus=$(IFS=,; echo "${GPU_LIST[*]:k},${GPU_LIST[*]:0:k}" | sed 's/,$//')
  if [[ -f "$OUT/$name/summary.md" ]]; then echo "skip $name (done)"; return; fi
  # shellcheck disable=SC2086
  "$PY" scripts/train_baselines.py data/contact_v1 --models B D --steps 20000 --eval_every 250 \
      --gpus "$gpus" --jobs_per_gpu 1 --examples 4 --out "$OUT/$name" $BASE ${CONFIGS[$name]} > "$OUT/$name.log" 2>&1 \
    && echo "done $name" || echo "FAILED $name (see $OUT/$name.log)"
}

i=0
for name in $(printf '%s\n' "${!CONFIGS[@]}" | sort); do
  while (( $(jobs -rp | wc -l) >= PARALLEL )); do sleep 5; done
  run "$name" $(( i * 6 )) &
  i=$(( i + 1 ))
done
wait
"$PY" scripts/compare_runs.py "$OUT"/*/ --out "$OUT"
