#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

# Total number of sweeps to keep per variant (including existing runs).
# Backward-compatible: SWEEP_COUNT can still be used.
TARGET_TOTAL_SWEEPS="${TARGET_TOTAL_SWEEPS:-${SWEEP_COUNT:-3}}"
ITERATIONS="${ITERATIONS:-100}"
NUM_TOP="${NUM_TOP:-3}"
NUM_DIVERSE="${NUM_DIVERSE:-2}"
NUM_INSPIRE="${NUM_INSPIRE:-3}"

GEMINI_KEY_FILE="${GEMINI_KEY_FILE:-keys/gemini-ge0.key}"
# Random 12-24 seed behavior:
# - per_round (default): seed = base + (round - 1)
# - fixed: always use the same seed across rounds
RANDOM_VARIANT_SEED_STRATEGY="${RANDOM_VARIANT_SEED_STRATEGY:-per_round}"
RANDOM_VARIANT_SAMPLE_SEED_BASE="${QOSE_RANDOM_SAMPLE_SEED_BASE:-${QOSE_RANDOM_SAMPLE_SEED:-123}}"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
}

require_file "$GEMINI_KEY_FILE"

next_output_dir() {
  local base="$1"
  local max_v=0
  local path name suffix version
  shopt -s nullglob
  for path in "${base}"_v*; do
    [[ -d "$path" ]] || continue
    name="$(basename "$path")"
    suffix="${name##*_v}"
    if [[ "$suffix" =~ ^[0-9]+$ ]]; then
      version="$suffix"
      if (( version > max_v )); then
        max_v="$version"
      fi
    fi
  done
  shopt -u nullglob
  echo "${base}_v$((max_v + 1))"
}

version_output_dir() {
  local base="$1"
  local round="$2"
  echo "${base}_v${round}"
}

version_completed() {
  local base="$1"
  local round="$2"
  local out_dir
  out_dir="$(version_output_dir "$base" "$round")"
  [[ -f "$out_dir/best/best_program_info.json" ]]
}

random_seed_for_round() {
  local round="$1"
  local base="$RANDOM_VARIANT_SAMPLE_SEED_BASE"
  local mode="${RANDOM_VARIANT_SEED_STRATEGY,,}"
  if [[ "$mode" == "fixed" ]]; then
    echo "$base"
    return 0
  fi
  # default: per_round
  echo $((base + round - 1))
}

run_variant() {
  local label="$1"
  local size_min="$2"
  local size_max="$3"
  local output_dir="$4"
  local sample_seed="${5:-}"

  echo "Running ${label} -> ${output_dir} (size_min=${size_min}, size_max=${size_max}, sample_seed=${sample_seed:-default})"

  local resume_latest="0"
  if [[ -d "$output_dir/checkpoints" ]]; then
    resume_latest="1"
  fi

  OPENAI_API_KEY="$(tr -d '\r\n' < "$GEMINI_KEY_FILE")" \
  GEMINI_RPM=0 \
  OPENEVOLVE_RPM=0 \
  RESUME_LATEST="$resume_latest" \
  QOSE_SIZE_MIN="$size_min" \
  QOSE_SIZE_MAX="$size_max" \
  QOSE_SAMPLES_PER_BENCH=1 \
  QOSE_SAMPLE_SEED="${sample_seed}" \
  QOSE_SCORE_MODE=piecewise \
  QOSE_INCLUDE_EXAMPLE_CODE=1 \
  QOSE_INCLUDE_CASES_ARTIFACT=1 \
  QOSE_INCLUDE_SUMMARY_ARTIFACT=1 \
  OPENAI_MODEL="gemini-3-flash-preview" \
  OPENEVOLVE_GEMINI_THINKING_LEVEL=low \
  OPENEVOLVE_DIFF_BASED_EVOLUTION=0 \
  ./run_oe.sh gemini "$output_dir" \
    --iterations "$ITERATIONS" \
    --num-top "$NUM_TOP" --num-diverse "$NUM_DIVERSE" --num-inspire "$NUM_INSPIRE"
}

run_variant_for_round() {
  local round="$1"
  local label="$2"
  local size_min="$3"
  local size_max="$4"
  local output_base="$5"
  local sample_seed="${6:-}"
  local output_dir
  output_dir="$(version_output_dir "$output_base" "$round")"

  if version_completed "$output_base" "$round"; then
    echo "Round ${round}: skip ${label} (completed: ${output_dir})"
    return 0
  fi

  echo "Round ${round}: run ${label} -> ${output_dir}"
  run_variant "$label" "$size_min" "$size_max" "$output_dir" "$sample_seed"
}

for round in $(seq 1 "$TARGET_TOTAL_SWEEPS"); do
  random_seed="$(random_seed_for_round "$round")"
  echo "Round ${round}: random-12to24 sample seed=${random_seed} (strategy=${RANDOM_VARIANT_SEED_STRATEGY})"

  run_variant_for_round \
    "$round" \
    "12q-only" \
    "12" "12" \
    "openevolve_ablation/gem3flash_pws8_12q_seed_low_full"

  run_variant_for_round \
    "$round" \
    "24q-only" \
    "24" "24" \
    "openevolve_ablation/gem3flash_pws8_24q_seed_low_full"

  run_variant_for_round \
    "$round" \
    "random-12to24" \
    "12" "24" \
    "openevolve_ablation/gem3flash_pws8_12to24_random_seed_low_full" \
    "$random_seed"
done
