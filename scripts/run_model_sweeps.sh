#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SWEEP_COUNT="${SWEEP_COUNT:-1}"
ITERATIONS="${ITERATIONS:-100}"
NUM_TOP="${NUM_TOP:-3}"
NUM_DIVERSE="${NUM_DIVERSE:-2}"
NUM_INSPIRE="${NUM_INSPIRE:-3}"

GEMINI_KEY_FILE="${GEMINI_KEY_FILE:-keys/gemini-ge0.key}"
OPENAI_KEY_FILE="${OPENAI_KEY_FILE:-keys/openai.key}"
CLAUDE_KEY_FILE="${CLAUDE_KEY_FILE:-keys/claude.key}"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
}

require_file "$GEMINI_KEY_FILE"
require_file "$OPENAI_KEY_FILE"
require_file "$CLAUDE_KEY_FILE"

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

run_gemini() {
  local model="$1"
  local thinking="$2"
  local include_example="$3"
  local output_dir="$4"
  local resume_latest="${5:-0}"

  OPENAI_API_KEY="$(tr -d '\r\n' < "$GEMINI_KEY_FILE")" \
  GEMINI_RPM=0 \
  OPENEVOLVE_RPM=0 \
  RESUME_LATEST="$resume_latest" \
  QOSE_SIZE_MIN=22 \
  QOSE_SIZE_MAX=22 \
  QOSE_SCORE_MODE=piecewise \
  QOSE_INCLUDE_EXAMPLE_CODE="$include_example" \
  OPENAI_MODEL="$model" \
  OPENEVOLVE_GEMINI_THINKING_LEVEL="$thinking" \
  OPENEVOLVE_DIFF_BASED_EVOLUTION=0 \
  ./run_oe.sh gemini "$output_dir" \
    --iterations "$ITERATIONS" \
    --num-top "$NUM_TOP" --num-diverse "$NUM_DIVERSE" --num-inspire "$NUM_INSPIRE"
}

run_gemini_for_round() {
  local round="$1"
  local model="$2"
  local thinking="$3"
  local include_example="$4"
  local output_base="$5"
  local output_dir
  output_dir="$(version_output_dir "$output_base" "$round")"
  if version_completed "$output_base" "$round"; then
    echo "Round ${round}: skip ${output_base} (completed: ${output_dir})"
    return 0
  fi
  local resume_latest="0"
  if [[ -d "$output_dir/checkpoints" ]]; then
    resume_latest="1"
    echo "Round ${round}: resume ${output_base} at ${output_dir}"
  else
    echo "Round ${round}: run ${output_base} -> ${output_dir}"
  fi
  run_gemini "$model" "$thinking" "$include_example" "$output_dir" "$resume_latest"
}

run_gpt() {
  local model="$1"
  local service_tier="$2"
  local output_dir="$3"
  local resume_latest="${4:-0}"

  OPENAI_API_KEY="$(tr -d '\r\n' < "$OPENAI_KEY_FILE")" \
  RESUME_LATEST="$resume_latest" \
  QOSE_SIZE_MIN=22 \
  QOSE_SIZE_MAX=22 \
  QOSE_SCORE_MODE=piecewise \
  QOSE_INCLUDE_EXAMPLE_CODE=1 \
  OPENEVOLVE_DIFF_BASED_EVOLUTION=0 \
  OPENAI_MODEL="$model" \
  OPENAI_SERVICE_TIER="$service_tier" \
  ./run_oe.sh gpt "$output_dir" \
    --iterations "$ITERATIONS" \
    --num-top "$NUM_TOP" --num-diverse "$NUM_DIVERSE" --num-inspire "$NUM_INSPIRE"
}

run_gpt_for_round() {
  local round="$1"
  local model="$2"
  local service_tier="$3"
  local output_base="$4"
  local output_dir
  output_dir="$(version_output_dir "$output_base" "$round")"
  if version_completed "$output_base" "$round"; then
    echo "Round ${round}: skip ${output_base} (completed: ${output_dir})"
    return 0
  fi
  local resume_latest="0"
  if [[ -d "$output_dir/checkpoints" ]]; then
    resume_latest="1"
    echo "Round ${round}: resume ${output_base} at ${output_dir}"
  else
    echo "Round ${round}: run ${output_base} -> ${output_dir}"
  fi
  run_gpt "$model" "$service_tier" "$output_dir" "$resume_latest"
}

run_claude() {
  local model="$1"
  local output_dir="$2"
  local resume_latest="${3:-0}"

  OPENAI_API_BASE="https://api.anthropic.com/v1/" \
  OPENAI_API_KEY="$(tr -d '\r\n' < "$CLAUDE_KEY_FILE")" \
  RESUME_LATEST="$resume_latest" \
  QOSE_SIZE_MIN=22 \
  QOSE_SIZE_MAX=22 \
  QOSE_SCORE_MODE=piecewise \
  QOSE_INCLUDE_EXAMPLE_CODE=1 \
  OPENEVOLVE_DIFF_BASED_EVOLUTION=0 \
  OPENAI_MODEL="$model" \
  ./run_oe.sh custom "$output_dir" \
    --iterations "$ITERATIONS" \
    --num-top "$NUM_TOP" --num-diverse "$NUM_DIVERSE" --num-inspire "$NUM_INSPIRE"
}

run_claude_for_round() {
  local round="$1"
  local model="$2"
  local output_base="$3"
  local output_dir
  output_dir="$(version_output_dir "$output_base" "$round")"
  if version_completed "$output_base" "$round"; then
    echo "Round ${round}: skip ${output_base} (completed: ${output_dir})"
    return 0
  fi
  local resume_latest="0"
  if [[ -d "$output_dir/checkpoints" ]]; then
    resume_latest="1"
    echo "Round ${round}: resume ${output_base} at ${output_dir}"
  else
    echo "Round ${round}: run ${output_base} -> ${output_dir}"
  fi
  run_claude "$model" "$output_dir" "$resume_latest"
}

for round in $(seq 1 "$SWEEP_COUNT"); do
  # Fixed order per round:
  # gpt5mini -> gpt53codex -> claude sonnet -> claude opus -> gemini pro -> gemini flash
  run_gpt_for_round "$round" "gpt-5-mini" "flex" \
    "openevolve_output/gpt5mini_pws8_22q_full"

  run_gpt_for_round "$round" "gpt-5.3-codex" "default" \
    "openevolve_output/gpt53codex_pws8_22q_full"

  run_claude_for_round "$round" "claude-sonnet-4-6" \
    "openevolve_output/claude_sonnet46_pws8_22q_full"

  run_claude_for_round "$round" "claude-opus-4-6" \
    "openevolve_output/claude_opus46_pws8_22q_full"

  run_gemini_for_round "$round" "gemini-3-pro-preview" "low" "1" \
    "openevolve_output/gem3pro_pws8_22q_seed_low_full"

  run_gemini_for_round "$round" "gemini-3-flash-preview" "low" "1" \
    "openevolve_output/gem3flash_pws8_22q_seed_low_full"
done
