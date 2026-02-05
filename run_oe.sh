#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_oe.sh <profile> <output_dir> [extra_args...]

Profiles:
  gpt     -> OpenAI-compatible (api.openai.com)
  gemini  -> Gemini OpenAI-compatible endpoint
  qwen    -> Local OpenAI-compatible endpoint (http://localhost:8000/v1)
  custom  -> Use existing OPENAI_API_BASE / OPENAI_MODEL / OPENAI_API_KEY

Environment variables (override defaults):
  OPENAI_API_BASE   API base URL
  OPENAI_MODEL      Model name
  OPENAI_API_KEY    API key (or set OPENAI_API_KEY_FILE)
  OPENAI_API_KEY_FILE  Path to file containing API key
  OPENEVOLVE_NUM_TOP_PROGRAMS     Override num_top_programs
  OPENEVOLVE_NUM_DIVERSE_PROGRAMS Override num_diverse_programs
  OPENEVOLVE_NUM_INSPIRATIONS     Override inspiration count
  OPENEVOLVE_INCLUDE_PEER_ARTIFACTS Enable peer (top/diverse/inspire) execution outputs
  RESUME_LATEST=1                 Resume from latest checkpoint under output_dir

Flags:
  --resume-latest                Resume from latest checkpoint under output_dir
  --target PATH                  Target program file (default: qos/error_mitigator/evolution_target.py)
  --repeat N                     Run N times (gemini only; rotates keys)
  --key-start N                  Starting key index for gemini (default 0)
  --key-prefix PATH              Key file prefix for gemini (default keys/gemini-ge)

Example:
  OPENAI_API_KEY=... ./run_oe.sh gpt openevolve_output/gpt_run --iterations 100
  OPENAI_API_KEY=... ./run_oe.sh gemini openevolve_output/gemini_run --iterations 100
  OPENAI_API_KEY=local ./run_oe.sh qwen openevolve_output/qwen_run --iterations 100
  OPENAI_API_BASE=... OPENAI_MODEL=... OPENAI_API_KEY=... ./run_oe.sh custom openevolve_output/custom_run
USAGE
}

if [[ $# -lt 2 ]]; then
  usage
  exit 1
fi

PROFILE="$1"
OUTPUT_DIR="$2"
shift 2

export OPENEVOLVE_NUM_TOP_PROGRAMS="${OPENEVOLVE_NUM_TOP_PROGRAMS:-3}"
export OPENEVOLVE_NUM_DIVERSE_PROGRAMS="${OPENEVOLVE_NUM_DIVERSE_PROGRAMS:-2}"
export OPENEVOLVE_NUM_INSPIRATIONS="${OPENEVOLVE_NUM_INSPIRATIONS:-3}"
REPEAT=1
KEY_START=0
KEY_PREFIX="keys/gemini-ge"

TARGET_PATH="qos/error_mitigator/evolution_target.py"
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --num-top)
      export OPENEVOLVE_NUM_TOP_PROGRAMS="$2"
      shift 2
      ;;
    --num-diverse)
      export OPENEVOLVE_NUM_DIVERSE_PROGRAMS="$2"
      shift 2
      ;;
    --num-inspire|--num-inspiration|--num-inspirations)
      export OPENEVOLVE_NUM_INSPIRATIONS="$2"
      shift 2
      ;;
    --memory)
      export OPENEVOLVE_INCLUDE_PEER_ARTIFACTS=1
      shift
      ;;
    --no-memory)
      export OPENEVOLVE_INCLUDE_PEER_ARTIFACTS=0
      shift
      ;;
    --resume-latest)
      export RESUME_LATEST=1
      shift
      ;;
    --target)
      TARGET_PATH="$2"
      shift 2
      ;;
    --repeat)
      REPEAT="$2"
      shift 2
      ;;
    --key-start)
      KEY_START="$2"
      shift 2
      ;;
    --key-prefix)
      KEY_PREFIX="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

# Default config uses env variables for base/model/key.
CONFIG_PATH="qos/error_mitigator/openevolve.yaml"

# Load API key from file if provided and OPENAI_API_KEY is unset.
if [[ -z "${OPENAI_API_KEY:-}" && -n "${OPENAI_API_KEY_FILE:-}" ]]; then
  if [[ -f "$OPENAI_API_KEY_FILE" ]]; then
    export OPENAI_API_KEY
    OPENAI_API_KEY="$(<"$OPENAI_API_KEY_FILE")"
  fi
fi

case "$PROFILE" in
  gpt)
    export OPENAI_API_BASE="${OPENAI_API_BASE:-https://api.openai.com/v1}"
    export OPENAI_MODEL="${OPENAI_MODEL:-gpt-5-mini}"
    export OPENAI_SERVICE_TIER="${OPENAI_SERVICE_TIER:-flex}"
    export OPENEVOLVE_PARALLEL_EVALUATIONS="${OPENEVOLVE_PARALLEL_EVALUATIONS:-1}"
    ;;
  gemini)
    export OPENAI_API_BASE="${OPENAI_API_BASE:-https://generativelanguage.googleapis.com/v1beta/openai/}"
    export OPENAI_MODEL="${OPENAI_MODEL:-gemini-2.5-flash-lite}"
    export GEMINI_RPM="${GEMINI_RPM:-5}"
    export OPENEVOLVE_PARALLEL_EVALUATIONS="${OPENEVOLVE_PARALLEL_EVALUATIONS:-1}"
    ;;
  qwen)
    export OPENAI_API_BASE="${OPENAI_API_BASE:-http://localhost:8000/v1}"
    export OPENAI_MODEL="${OPENAI_MODEL:-Qwen/Qwen2.5-Coder-14B-Instruct-AWQ}"
    # For local servers that don't require a key.
    export OPENAI_API_KEY="${OPENAI_API_KEY:-local}"
    export OPENEVOLVE_PARALLEL_EVALUATIONS="${OPENEVOLVE_PARALLEL_EVALUATIONS:-1}"
    ;;
  custom)
    if [[ -z "${OPENAI_API_BASE:-}" || -z "${OPENAI_MODEL:-}" || -z "${OPENAI_API_KEY:-}" ]]; then
      echo "custom profile requires OPENAI_API_BASE, OPENAI_MODEL, and OPENAI_API_KEY" >&2
      exit 1
    fi
    export OPENEVOLVE_PARALLEL_EVALUATIONS="${OPENEVOLVE_PARALLEL_EVALUATIONS:-1}"
    ;;
  *)
    echo "Unknown profile: $PROFILE" >&2
    usage
    exit 1
    ;;
 esac

if [[ "$REPEAT" -lt 1 ]]; then
  REPEAT=1
fi

run_once() {
  local -a run_args=("${EXTRA_ARGS[@]}")
  local run_idx="$1"
  if [[ "$run_idx" -gt 0 ]]; then
    latest_checkpoint="$(ls -d "$OUTPUT_DIR"/checkpoints/checkpoint_* 2>/dev/null | sort -V | tail -n 1)"
    if [[ -n "$latest_checkpoint" ]]; then
      echo "Resuming from latest checkpoint: $latest_checkpoint"
      run_args+=(--checkpoint "$latest_checkpoint")
    else
      echo "Resume requested, but no checkpoints found under $OUTPUT_DIR" >&2
    fi
  elif [[ "${RESUME_LATEST:-}" == "1" ]]; then
    latest_checkpoint="$(ls -d "$OUTPUT_DIR"/checkpoints/checkpoint_* 2>/dev/null | sort -V | tail -n 1)"
    if [[ -n "$latest_checkpoint" ]]; then
      echo "Resuming from latest checkpoint: $latest_checkpoint"
      run_args+=(--checkpoint "$latest_checkpoint")
    else
      echo "Resume requested, but no checkpoints found under $OUTPUT_DIR" >&2
    fi
  fi

  echo "Using target: $TARGET_PATH"
  conda run -n quantum python -m qos.error_mitigator.run_openevolve_rate_limited \
    "$TARGET_PATH" \
    qos/error_mitigator/evaluator.py \
    --config "$CONFIG_PATH" \
    --output "$OUTPUT_DIR" \
    "${run_args[@]}"
}

if [[ "$PROFILE" == "gemini" && "$REPEAT" -gt 1 ]]; then
  for ((i=0; i<REPEAT; i++)); do
    key_index=$((KEY_START + i))
    key_file="${KEY_PREFIX}${key_index}.key"
    if [[ -f "$key_file" ]]; then
      export OPENAI_API_KEY
      OPENAI_API_KEY="$(<"$key_file")"
      echo "Using gemini key: $key_file"
    else
      echo "Gemini key not found: $key_file" >&2
    fi
    run_once "$i"
  done
else
  run_once 0
fi
