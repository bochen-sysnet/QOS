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

run_gemini() {
  local model="$1"
  local thinking="$2"
  local include_example="$3"
  local output_base="$4"
  local output_dir
  output_dir="$(next_output_dir "$output_base")"

  OPENAI_API_KEY="$(tr -d '\r\n' < "$GEMINI_KEY_FILE")" \
  GEMINI_RPM=0 \
  OPENEVOLVE_RPM=0 \
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

run_gpt() {
  local model="$1"
  local service_tier="$2"
  local output_base="$3"
  local output_dir
  output_dir="$(next_output_dir "$output_base")"

  OPENAI_API_KEY="$(tr -d '\r\n' < "$OPENAI_KEY_FILE")" \
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

run_claude() {
  local model="$1"
  local output_base="$2"
  local output_dir
  output_dir="$(next_output_dir "$output_base")"

  OPENAI_API_BASE="https://api.anthropic.com/v1/" \
  OPENAI_API_KEY="$(tr -d '\r\n' < "$CLAUDE_KEY_FILE")" \
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

for sweep in $(seq 1 "$SWEEP_COUNT"); do
  run_gpt "gpt-5-mini" "flex" \
    "openevolve_output/gpt5mini_pws8_22q_full"

  run_gpt "gpt-5.3-codex" "default" \
    "openevolve_output/gpt53codex_pws8_22q_full"

  run_claude "claude-sonnet-4-6" \
    "openevolve_output/claude_sonnet46_pws8_22q_full"

  run_claude "claude-opus-4-6" \
    "openevolve_output/claude_opus46_pws8_22q_full"

  run_gemini "gemini-3-pro-preview" "low" "1" \
    "openevolve_output/gem3pro_pws8_22q_seed_low_full"

  run_gemini "gemini-3-flash-preview" "low" "1" \
    "openevolve_output/gem3flash_pws8_22q_seed_low_full"
done
