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
REFERENCE_FULL_PREFIX="${REFERENCE_FULL_PREFIX:-openevolve_output/gem3flash_pws8_22q_seed_low_full_v}"
ARTIFACT_OUT_DIR="${ARTIFACT_OUT_DIR:-openevolve_ablation/artifact}"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
}

require_file "$GEMINI_KEY_FILE"

latest_matching_dir() {
  local prefix="$1"
  local path best="" best_v=-1
  shopt -s nullglob
  for path in "${prefix}"*; do
    [[ -d "$path" ]] || continue
    if [[ "$path" =~ _v([0-9]+)$ ]]; then
      local version="${BASH_REMATCH[1]}"
      if (( version > best_v )); then
        best="$path"
        best_v="$version"
      fi
    fi
  done
  shopt -u nullglob
  if [[ -n "$best" ]]; then
    echo "$best"
  fi
}

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

latest_version_num() {
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
  echo "$max_v"
}

run_variant() {
  local variant_name="$1"
  local include_example="$2"
  local include_cases="$3"
  local include_summary="$4"
  local output_dir="$5"

  echo "Running ${variant_name} -> ${output_dir}"

  local resume_latest="0"
  if [[ -d "$output_dir/checkpoints" ]]; then
    resume_latest="1"
  fi

  OPENAI_API_KEY="$(tr -d '\r\n' < "$GEMINI_KEY_FILE")" \
  GEMINI_RPM=0 \
  OPENEVOLVE_RPM=0 \
  RESUME_LATEST="$resume_latest" \
  QOSE_SIZE_MIN=22 \
  QOSE_SIZE_MAX=22 \
  QOSE_SCORE_MODE=piecewise \
  QOSE_INCLUDE_EXAMPLE_CODE="$include_example" \
  QOSE_INCLUDE_CASES_ARTIFACT="$include_cases" \
  QOSE_INCLUDE_SUMMARY_ARTIFACT="$include_summary" \
  OPENAI_MODEL="gemini-3-flash-preview" \
  OPENEVOLVE_GEMINI_THINKING_LEVEL=low \
  OPENEVOLVE_DIFF_BASED_EVOLUTION=0 \
  bash ./run_oe.sh gemini "$output_dir" \
    --iterations "$ITERATIONS" \
    --num-top "$NUM_TOP" --num-diverse "$NUM_DIVERSE" --num-inspire "$NUM_INSPIRE"
}

run_variant_for_round() {
  local round="$1"
  local variant_name="$2"
  local include_example="$3"
  local include_cases="$4"
  local include_summary="$5"
  local output_base="$6"
  local output_dir
  output_dir="$(version_output_dir "$output_base" "$round")"

  if version_completed "$output_base" "$round"; then
    echo "Round ${round}: skip ${variant_name} (completed: ${output_dir})"
    return 0
  fi

  echo "Round ${round}: run ${variant_name} -> ${output_dir}"
  run_variant "$variant_name" "$include_example" "$include_cases" "$include_summary" "$output_dir"
}

reference_full="$(latest_matching_dir "$REFERENCE_FULL_PREFIX")"
if [[ -n "${reference_full:-}" ]]; then
  echo "Reference full run: ${reference_full}"
else
  echo "Reference full run not found under prefix: ${REFERENCE_FULL_PREFIX}" >&2
fi

for round in $(seq 1 "$TARGET_TOTAL_SWEEPS"); do
  # Required execution order per round:
  # 111 -> 110 -> 100
  run_variant_for_round \
    "$round" \
    "111_full_without_seed" \
    "0" "1" "1" \
    "${ARTIFACT_OUT_DIR}/gem3flash_pws8_22q_noseed_full"

  run_variant_for_round \
    "$round" \
    "110_full_without_seed_no_cases" \
    "0" "0" "1" \
    "${ARTIFACT_OUT_DIR}/gem3flash_pws8_22q_noseed_no_cases"

  run_variant_for_round \
    "$round" \
    "100_full_without_seed_no_cases_no_summary" \
    "0" "0" "0" \
    "${ARTIFACT_OUT_DIR}/gem3flash_pws8_22q_noseed_no_cases_no_summary"
done
