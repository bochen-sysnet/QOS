#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

# Total number of sweeps to keep per variant (including existing runs).
# Backward-compatible: SWEEP_COUNT can still be used.
TARGET_TOTAL_SWEEPS="${TARGET_TOTAL_SWEEPS:-${SWEEP_COUNT:-1}}"
ITERATIONS="${ITERATIONS:-100}"
NUM_TOP="${NUM_TOP:-3}"
NUM_DIVERSE="${NUM_DIVERSE:-2}"
NUM_INSPIRE="${NUM_INSPIRE:-3}"

GEMINI_KEY_FILE="${GEMINI_KEY_FILE:-keys/gemini-ge0.key}"
THINKING_OUT_DIR="${THINKING_OUT_DIR:-openevolve_ablation/thinking}"

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

run_variant() {
  local thinking="$1"
  local output_dir="$2"
  local resume_latest="${3:-0}"

  echo "Running gemini-3-flash-preview thinking=${thinking} -> ${output_dir}"

  OPENAI_API_KEY="$(tr -d '\r\n' < "$GEMINI_KEY_FILE")" \
  GEMINI_RPM=0 \
  OPENEVOLVE_RPM=0 \
  RESUME_LATEST="$resume_latest" \
  QOSE_SIZE_MIN=22 \
  QOSE_SIZE_MAX=22 \
  QOSE_SCORE_MODE=piecewise \
  QOSE_INCLUDE_EXAMPLE_CODE=1 \
  QOSE_INCLUDE_CASES_ARTIFACT=1 \
  QOSE_INCLUDE_SUMMARY_ARTIFACT=1 \
  OPENAI_MODEL="gemini-3-flash-preview" \
  OPENEVOLVE_GEMINI_THINKING_LEVEL="$thinking" \
  OPENEVOLVE_DIFF_BASED_EVOLUTION=0 \
  ./run_oe.sh gemini "$output_dir" \
    --iterations "$ITERATIONS" \
    --num-top "$NUM_TOP" --num-diverse "$NUM_DIVERSE" --num-inspire "$NUM_INSPIRE"
}

run_variant_for_round() {
  local round="$1"
  local thinking="$2"
  local output_base="$3"
  local output_dir
  output_dir="$(version_output_dir "$output_base" "$round")"

  if version_completed "$output_base" "$round"; then
    echo "Round ${round}: skip thinking=${thinking} (completed: ${output_dir})"
    return 0
  fi

  local resume_latest="0"
  if [[ -d "$output_dir/checkpoints" ]]; then
    resume_latest="1"
    echo "Round ${round}: resume thinking=${thinking} at ${output_dir}"
  else
    echo "Round ${round}: run thinking=${thinking} -> ${output_dir}"
  fi

  run_variant "$thinking" "$output_dir" "$resume_latest"
}

for sweep in $(seq 1 "$TARGET_TOTAL_SWEEPS"); do
  run_variant_for_round \
    "$sweep" \
    "medium" \
    "${THINKING_OUT_DIR}/gem3flash_pws8_22q_seed_medium_full"

  run_variant_for_round \
    "$sweep" \
    "high" \
    "${THINKING_OUT_DIR}/gem3flash_pws8_22q_seed_high_full"
done
