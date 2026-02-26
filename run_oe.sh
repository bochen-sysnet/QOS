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
  QOSE_SCORE_MODE                 Score profile: legacy (default) or piecewise
  OPENEVOLVE_CONFIG_PATH          Optional explicit config YAML path (overrides score-profile selection)
  RESUME_LATEST=1                 Resume from latest checkpoint under output_dir
  QOSE_INCLUDE_EXAMPLE_CODE       If true, injects example evolution code into prompt system message (default: 0)
  QOSE_EXAMPLE_CODE_PATH          Optional path for example code injection (default: qos/error_mitigator/evolution_seed.py)
  QOSE_SURROGATE_STATE_CSV        Surrogate cache path (default: <output_dir>/qose_surrogate_state.csv)
  QOSE_FIXED_BENCH_SIZE_PAIRS     Optional fixed sampled pairs (JSON list), e.g. [["qaoa_r3",22],["bv",20]]
  OPENEVOLVE_GEMINI_NATIVE        Use Gemini native generateContent API for gemini endpoint (default: 1)
  OPENEVOLVE_GEMINI_THINKING_LEVEL Optional Gemini thinking level: low|medium|high|auto (default: auto)
  OPENEVOLVE_GEMINI_MAX_OUTPUT_TOKENS  Override maxOutputTokens for Gemini native calls
  OPENAI_REASONING_EFFORT         Optional GPT-5 reasoning effort: minimal|low|medium|high|auto

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

# Default surrogate cache location: per-evolution output directory.
# This keeps correlation/prediction state isolated by run unless explicitly overridden.
if [[ -z "${QOSE_SURROGATE_STATE_CSV:-}" ]]; then
  export QOSE_SURROGATE_STATE_CSV="$OUTPUT_DIR/qose_surrogate_state.csv"
fi

# Default: keep surrogate disabled unless explicitly enabled.
export QOSE_SURROGATE_ENABLE="${QOSE_SURROGATE_ENABLE:-0}"

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
# Score profile is selected by QOSE_SCORE_MODE:
# - piecewise -> openevolve_piecewise.yaml
# - legacy    -> openevolve_legacy.yaml (default)
SCORE_MODE_RAW="${QOSE_SCORE_MODE:-legacy}"
SCORE_MODE_NORM="$(echo "$SCORE_MODE_RAW" | tr '[:upper:]' '[:lower:]')"
case "$SCORE_MODE_NORM" in
  piecewise|pwl|piecewise_linear)
    CONFIG_PATH="qos/error_mitigator/openevolve_piecewise.yaml"
    export QOSE_SCORE_MODE="piecewise"
    ;;
  legacy|"")
    CONFIG_PATH="qos/error_mitigator/openevolve_legacy.yaml"
    export QOSE_SCORE_MODE="legacy"
    ;;
  *)
    echo "Unknown QOSE_SCORE_MODE='$SCORE_MODE_RAW'; using legacy config." >&2
    CONFIG_PATH="qos/error_mitigator/openevolve_legacy.yaml"
    export QOSE_SCORE_MODE="legacy"
    ;;
esac

# Optional explicit config override.
if [[ -n "${OPENEVOLVE_CONFIG_PATH:-}" ]]; then
  CONFIG_PATH="$OPENEVOLVE_CONFIG_PATH"
fi

is_true() {
  local raw="${1:-}"
  raw="$(echo "$raw" | tr '[:upper:]' '[:lower:]')"
  [[ "$raw" == "1" || "$raw" == "true" || "$raw" == "yes" || "$raw" == "y" ]]
}

build_runtime_config() {
  local base_config_path="$1"
  local runtime_config_path="$2"
  local include_example_raw="${QOSE_INCLUDE_EXAMPLE_CODE:-0}"
  local example_path="${QOSE_EXAMPLE_CODE_PATH:-qos/error_mitigator/evolution_seed.py}"

  if ! is_true "$include_example_raw"; then
    cp -f "$base_config_path" "$runtime_config_path"
    return
  fi

  if [[ ! -f "$example_path" ]]; then
    echo "QOSE_INCLUDE_EXAMPLE_CODE is enabled but example file was not found: $example_path" >&2
    echo "Proceeding without example injection." >&2
    cp -f "$base_config_path" "$runtime_config_path"
    return
  fi

  python3 - "$base_config_path" "$runtime_config_path" "$example_path" <<'PY'
import re
import sys
from pathlib import Path

base_cfg = Path(sys.argv[1])
runtime_cfg = Path(sys.argv[2])
example_path = Path(sys.argv[3])

text = base_cfg.read_text(encoding="utf-8")
example_code = example_path.read_text(encoding="utf-8").rstrip("\n")

# Keep config untouched if already injected.
if "Example Evolution:" in text:
    runtime_cfg.write_text(text, encoding="utf-8")
    raise SystemExit(0)

# Prefer text-level insertion to preserve comments/formatting of the original YAML.
pattern = re.compile(
    r"(^  system_message:\s*\|\n)(.*?)(?=^  [A-Za-z_][A-Za-z0-9_]*\s*:)",
    re.MULTILINE | re.DOTALL,
)
match = pattern.search(text)
if not match:
    # Fallback: do not rewrite into escaped YAML if format is unexpected.
    runtime_cfg.write_text(text, encoding="utf-8")
    raise SystemExit(0)

body = match.group(2).rstrip("\n")
example_lines = [
    "",
    "    Example Evolution:",
    "    ```python",
]
example_lines += [f"    {line}" if line else "    " for line in example_code.splitlines()]
example_lines.append("    ```")
new_body = body + "\n" + "\n".join(example_lines) + "\n"

patched = text[: match.start(2)] + new_body + text[match.end(2) :]
runtime_cfg.write_text(patched, encoding="utf-8")
PY
}

# Keep a runtime copy of evolve config in the output directory for reproducibility.
mkdir -p "$OUTPUT_DIR"
RUNTIME_CONFIG_PATH="$OUTPUT_DIR/openevolve_config.yaml"
build_runtime_config "$CONFIG_PATH" "$RUNTIME_CONFIG_PATH"
CONFIG_PATH="$RUNTIME_CONFIG_PATH"

# Load API key from file if provided and OPENAI_API_KEY is unset.
if [[ -z "${OPENAI_API_KEY:-}" && -n "${OPENAI_API_KEY_FILE:-}" ]]; then
  if [[ -f "$OPENAI_API_KEY_FILE" ]]; then
    export OPENAI_API_KEY
    OPENAI_API_KEY="$(tr -d '\r\n' < "$OPENAI_API_KEY_FILE")"
  fi
fi

load_key_from_file() {
  local key_file="$1"
  if [[ ! -f "$key_file" ]]; then
    return 1
  fi
  export OPENAI_API_KEY
  OPENAI_API_KEY="$(tr -d '\r\n' < "$key_file")"
  [[ -n "${OPENAI_API_KEY:-}" ]]
}

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
    export OPENEVOLVE_GEMINI_THINKING_LEVEL="${OPENEVOLVE_GEMINI_THINKING_LEVEL:-auto}"
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

record_run_environment() {
  local ts env_path
  ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  env_path="$OUTPUT_DIR/run_env_vars.log"
  {
    echo "[$ts]"
    env | sort | grep -E '^(QOSE_|OPENEVOLVE_|OPENAI_|GEMINI_|QISKIT_|IBM_)|^RESUME_LATEST=' || true
    echo
  } >> "$env_path"
}

record_run_environment

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
    if load_key_from_file "$key_file"; then
      echo "Using gemini key: $key_file"
    else
      echo "Gemini key missing/empty: $key_file" >&2
      exit 1
    fi
    run_once "$i"
  done
else
  if [[ "$PROFILE" == "gemini" && -z "${OPENAI_API_KEY:-}" ]]; then
    key_file="${KEY_PREFIX}${KEY_START}.key"
    if load_key_from_file "$key_file"; then
      echo "Using gemini key: $key_file"
    else
      echo "OPENAI_API_KEY is not set and Gemini key missing/empty: $key_file" >&2
      exit 1
    fi
  fi
  if [[ "$PROFILE" == "gpt" && -z "${OPENAI_API_KEY:-}" ]]; then
    echo "OPENAI_API_KEY is not set for gpt profile." >&2
    exit 1
  fi
  run_once 0
fi
