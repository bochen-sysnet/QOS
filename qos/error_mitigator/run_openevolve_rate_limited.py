import asyncio
import csv
import datetime
import fcntl
import json
import logging
import math
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from openevolve.cli import main as openevolve_main
from openevolve.llm.openai import OpenAILLM
from openevolve.prompt.sampler import PromptSampler
from openevolve.database import ProgramDatabase, Program
from openevolve.evaluator import Evaluator
from openevolve.utils.metrics_utils import safe_numeric_average


_GEMINI_API_PREFIX = "https://generativelanguage.googleapis.com/"
_OPENAI_API_PREFIX = "https://api.openai.com/"
_ANTHROPIC_API_PREFIX = "https://api.anthropic.com/"
_PROMPT_ARTIFACTS_PATCHED = False
_PROMPT_LOGGING_PATCHED = False
_EVALUATOR_ARTIFACTS_PATCHED = False
_DATABASE_ARTIFACTS_PATCHED = False
_PENDING_ARTIFACTS: dict[str, dict] = {}
_PROCESS_PARALLEL_PATCHED = False
_PROMPT_CONFIG_PATCHED = False
_LLM_CONFIG_PATCHED = False
_EVALUATOR_CONFIG_PATCHED = False
_RUNTIME_TRACKING_PATCHED = False
_logger = logging.getLogger(__name__)

_RUNTIME_ITERATION_CSV_PATH = ""
_RUNTIME_TRACKING_READY = False
_USAGE_WARN_LOCK = threading.Lock()
_USAGE_WARNED: set[tuple[str, str, str, str]] = set()


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()


def _estimate_tokens_from_text(text: str) -> int:
    if not text:
        return 0
    # Lightweight approximation used when provider token accounting is unavailable.
    return max(1, int(math.ceil(len(text) / 4.0)))


def _warn_missing_usage_once(
    *, api_base: str, model: str, source: str, missing_fields: list[str], available_fields: list[str]
) -> None:
    if not missing_fields:
        return
    key = (api_base or "", model or "", source or "", ",".join(sorted(missing_fields)))
    with _USAGE_WARN_LOCK:
        if key in _USAGE_WARNED:
            return
        _USAGE_WARNED.add(key)
    _logger.warning(
        "Could not extract expected usage fields from response (source=%s, api_base=%s, model=%s). "
        "missing=%s available=%s",
        source,
        api_base or "",
        model or "",
        ",".join(missing_fields),
        ",".join(available_fields),
    )


def _cli_arg_value(flag: str) -> str:
    for idx, arg in enumerate(sys.argv):
        if arg == flag and idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return ""


def _init_runtime_tracking_paths() -> None:
    global _RUNTIME_TRACKING_READY
    global _RUNTIME_ITERATION_CSV_PATH
    if _RUNTIME_TRACKING_READY:
        return
    # Always prefer explicit CLI --output over inherited environment variables.
    cli_output_dir = _cli_arg_value("--output").strip()
    output_dir = cli_output_dir or os.getenv("OPENEVOLVE_OUTPUT_DIR", "").strip()
    if cli_output_dir:
        os.environ["OPENEVOLVE_OUTPUT_DIR"] = cli_output_dir
    if not output_dir:
        return
    os.makedirs(output_dir, exist_ok=True)
    _RUNTIME_ITERATION_CSV_PATH = os.path.join(output_dir, "runtime_metrics.iterations.csv")
    _RUNTIME_TRACKING_READY = True


_RUNTIME_ITERATION_CSV_FIELDS = [
    "ts",
    "run_index",
    "pid",
    "iteration",
    "success",
    "stage",
    "llm_elapsed_time_sec",
    "evaluation_elapsed_time_sec",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cached_tokens",
    "reasoning_tokens",
    "combined_score",
    "qose_depth",
    "qose_cnot",
    "qose_overhead",
    "avg_run_time",
    "error",
]


def _next_run_index_unlocked() -> int:
    if not _RUNTIME_ITERATION_CSV_PATH or not os.path.exists(_RUNTIME_ITERATION_CSV_PATH):
        return 1
    max_run_index = 0
    try:
        with open(_RUNTIME_ITERATION_CSV_PATH, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                raw = str(row.get("run_index", "")).strip()
                if not raw:
                    continue
                try:
                    max_run_index = max(max_run_index, int(raw))
                except ValueError:
                    continue
    except Exception:
        return 1
    return max_run_index + 1


def _ensure_iteration_csv_header_unlocked() -> None:
    if not _RUNTIME_ITERATION_CSV_PATH:
        return
    if os.path.exists(_RUNTIME_ITERATION_CSV_PATH) and os.path.getsize(_RUNTIME_ITERATION_CSV_PATH) > 0:
        return
    with open(_RUNTIME_ITERATION_CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_RUNTIME_ITERATION_CSV_FIELDS)
        writer.writeheader()


def _append_iteration_csv_unlocked(event: dict) -> None:
    if not _RUNTIME_ITERATION_CSV_PATH:
        return
    payload = event.get("payload") or {}
    usage = payload.get("reported_usage") or {}
    if not isinstance(usage, dict):
        usage = {}
    _ensure_iteration_csv_header_unlocked()
    row = {
        "ts": event.get("ts", ""),
        "run_index": payload.get("run_index", ""),
        "pid": event.get("pid", ""),
        "iteration": payload.get("iteration", ""),
        "success": int(bool(payload.get("success", False))),
        "stage": payload.get("stage", ""),
        "llm_elapsed_time_sec": float(payload.get("llm_phase_sec", 0.0) or 0.0),
        "evaluation_elapsed_time_sec": float(payload.get("evaluation_phase_sec", 0.0) or 0.0),
        "prompt_tokens": float(usage.get("prompt_tokens", 0.0) or 0.0),
        "completion_tokens": float(usage.get("completion_tokens", 0.0) or 0.0),
        "total_tokens": float(usage.get("total_tokens", 0.0) or 0.0),
        "cached_tokens": float(
            usage.get("cached_tokens", usage.get("input_tokens_details.cached_tokens", 0.0)) or 0.0
        ),
        "reasoning_tokens": float(
            usage.get(
                "reasoning_tokens",
                usage.get("output_tokens_details.reasoning_tokens", 0.0),
            )
            or 0.0
        ),
        "combined_score": payload.get("combined_score", ""),
        "qose_depth": payload.get("qose_depth", ""),
        "qose_cnot": payload.get("qose_cnot", ""),
        "qose_overhead": payload.get("qose_overhead", ""),
        "avg_run_time": payload.get("avg_run_time", ""),
        "error": payload.get("error", ""),
    }
    with open(_RUNTIME_ITERATION_CSV_PATH, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_RUNTIME_ITERATION_CSV_FIELDS)
        writer.writerow(row)


def _record_runtime_event(event_type: str, payload: dict) -> None:
    _init_runtime_tracking_paths()
    if not _RUNTIME_TRACKING_READY:
        return
    payload = dict(payload or {})
    event = {
        "ts": _utc_now_iso(),
        "type": event_type,
        "pid": os.getpid(),
        "payload": payload,
    }
    os.makedirs(os.path.dirname(_RUNTIME_ITERATION_CSV_PATH), exist_ok=True)
    with open(_RUNTIME_ITERATION_CSV_PATH, "a+", encoding="utf-8", newline="") as csv_lockf:
        fcntl.flock(csv_lockf.fileno(), fcntl.LOCK_EX)
        try:
            if event_type == "run_start":
                run_index = _next_run_index_unlocked()
                payload["run_index"] = run_index
                event["payload"] = payload
                os.environ["OPENEVOLVE_RUNTIME_RUN_INDEX"] = str(run_index)
            elif "run_index" not in payload:
                run_index = os.getenv("OPENEVOLVE_RUNTIME_RUN_INDEX", "").strip()
                if run_index:
                    payload["run_index"] = run_index
                    event["payload"] = payload
            if event_type == "iteration":
                _append_iteration_csv_unlocked(event)
        finally:
            fcntl.flock(csv_lockf.fileno(), fcntl.LOCK_UN)


def _get_inspiration_limit(config) -> int:
    env_value = os.getenv("OPENEVOLVE_NUM_INSPIRATIONS")
    if env_value is not None:
        try:
            return max(0, int(env_value))
        except ValueError:
            return 3
    return 3


def _install_prompt_config_overrides() -> None:
    global _PROMPT_CONFIG_PATCHED
    if _PROMPT_CONFIG_PATCHED:
        return
    try:
        from openevolve import config as oe_config
    except Exception:
        return

    original_init = oe_config.PromptConfig.__init__

    def wrapped_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        num_top = os.getenv("OPENEVOLVE_NUM_TOP_PROGRAMS")
        num_diverse = os.getenv("OPENEVOLVE_NUM_DIVERSE_PROGRAMS")
        if num_top is not None:
            try:
                self.num_top_programs = max(0, int(num_top))
            except ValueError:
                pass
        if num_diverse is not None:
            try:
                self.num_diverse_programs = max(0, int(num_diverse))
            except ValueError:
                pass

    oe_config.PromptConfig.__init__ = wrapped_init
    _PROMPT_CONFIG_PATCHED = True


def _install_llm_config_overrides() -> None:
    global _LLM_CONFIG_PATCHED
    if _LLM_CONFIG_PATCHED:
        return
    try:
        from openevolve import config as oe_config
    except Exception:
        return

    original_post_init = oe_config.LLMConfig.__post_init__

    def wrapped_post_init(self):
        try:
            self.api_base = oe_config._resolve_env_var(self.api_base)
            self.primary_model = oe_config._resolve_env_var(self.primary_model)
            self.secondary_model = oe_config._resolve_env_var(self.secondary_model)
        except ValueError:
            pass
        return original_post_init(self)

    oe_config.LLMConfig.__post_init__ = wrapped_post_init
    _LLM_CONFIG_PATCHED = True


def _install_evaluator_config_overrides() -> None:
    global _EVALUATOR_CONFIG_PATCHED
    if _EVALUATOR_CONFIG_PATCHED:
        return
    try:
        from openevolve import config as oe_config
    except Exception:
        return

    original_init = oe_config.EvaluatorConfig.__init__

    def wrapped_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        env_value = os.getenv("OPENEVOLVE_PARALLEL_EVALUATIONS")
        if env_value:
            try:
                self.parallel_evaluations = max(1, int(env_value))
            except ValueError:
                pass

    oe_config.EvaluatorConfig.__init__ = wrapped_init
    _EVALUATOR_CONFIG_PATCHED = True


def _patched_run_iteration_worker(
    iteration: int, db_snapshot: dict, parent_id: str, inspiration_ids: list
):
    import openevolve.process_parallel as process_parallel

    logger = logging.getLogger("openevolve.process_parallel")
    try:
        process_parallel._lazy_init_worker_components()
        prompt_build_sec = 0.0
        llm_phase_sec = 0.0
        code_phase_sec = 0.0
        eval_phase_sec = 0.0
        iteration_reported_usage: dict[str, float] = {}

        programs = {
            pid: Program(**prog_dict)
            for pid, prog_dict in db_snapshot["programs"].items()
        }

        parent = programs[parent_id]
        inspirations = [programs[pid] for pid in inspiration_ids if pid in programs]

        parent_artifacts = db_snapshot["artifacts"].get(parent_id)

        parent_island = parent.metadata.get("island", db_snapshot["current_island"])
        island_programs = [
            programs[pid]
            for pid in db_snapshot["islands"][parent_island]
            if pid in programs
        ]

        island_programs.sort(
            key=lambda p: p.metrics.get("combined_score", safe_numeric_average(p.metrics)),
            reverse=True,
        )

        # Use all island programs as the prompt pool so we can select distinct inspirations.
        programs_for_prompt = island_programs
        best_programs_only = island_programs[
            : process_parallel._worker_config.prompt.num_top_programs
        ]

        t_prompt_build = time.perf_counter()
        prompt = process_parallel._worker_prompt_sampler.build_prompt(
            current_program=parent.code,
            parent_program=parent.code,
            program_metrics=parent.metrics,
            previous_programs=[p.to_dict() for p in best_programs_only],
            top_programs=[p.to_dict() for p in programs_for_prompt],
            inspirations=[p.to_dict() for p in inspirations],
            language=process_parallel._worker_config.language,
            evolution_round=iteration,
            diff_based_evolution=process_parallel._worker_config.diff_based_evolution,
            program_artifacts=parent_artifacts,
            feature_dimensions=db_snapshot.get("feature_dimensions", []),
        )
        prompt_build_sec = time.perf_counter() - t_prompt_build

        iteration_start = time.time()
        llm_success = False
        eval_success = False

        def _consume_worker_llm_usage() -> dict[str, float]:
            usage_totals: dict[str, float] = {}
            ensemble = getattr(process_parallel, "_worker_llm_ensemble", None)
            for model in getattr(ensemble, "models", []) or []:
                fields = getattr(model, "_oe_last_usage_fields", {}) or {}
                if isinstance(fields, dict):
                    for key, value in fields.items():
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            usage_totals[key] = float(usage_totals.get(key, 0.0)) + float(value)
                setattr(model, "_oe_last_usage_fields", {})
            return usage_totals

        def _record_iteration_runtime(
            success: bool,
            stage: str = "",
            error: str = "",
            combined_score=None,
            child_metrics: dict | None = None,
        ) -> None:
            payload = {
                "success": bool(success),
                "iteration": int(iteration),
                "iteration_wall_time_sec": time.time() - iteration_start,
                "prompt_build_sec": prompt_build_sec,
                "llm_phase_sec": llm_phase_sec,
                "code_phase_sec": code_phase_sec,
                "evaluation_phase_sec": eval_phase_sec,
                "reported_usage": dict(iteration_reported_usage),
            }
            if stage:
                payload["stage"] = stage
            if error:
                payload["error"] = error
            if combined_score is not None:
                payload["combined_score"] = combined_score
            if isinstance(child_metrics, dict):
                for key in ("qose_depth", "qose_cnot", "qose_overhead", "avg_run_time"):
                    if key in child_metrics:
                        payload[key] = child_metrics.get(key)
            _record_runtime_event("iteration", payload)

        _consume_worker_llm_usage()

        try:
            t_llm = time.perf_counter()
            llm_response = asyncio.run(
                process_parallel._worker_llm_ensemble.generate_with_context(
                    system_message=prompt["system"],
                    messages=[{"role": "user", "content": prompt["user"]}],
                )
            )
            llm_phase_sec = time.perf_counter() - t_llm
            llm_success = True
            iteration_reported_usage = _consume_worker_llm_usage()
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            _record_iteration_runtime(False, stage="llm", error=str(e))
            return process_parallel.SerializableResult(
                error=f"LLM generation failed: {str(e)}", iteration=iteration
            )

        if llm_response is None:
            _record_iteration_runtime(False, stage="llm_none")
            return process_parallel.SerializableResult(
                error="LLM returned None response", iteration=iteration
            )

        t_code_phase = time.perf_counter()
        if process_parallel._worker_config.diff_based_evolution:
            from openevolve.utils.code_utils import (
                apply_diff,
                extract_diffs,
                format_diff_summary,
            )

            diff_blocks = extract_diffs(
                llm_response, process_parallel._worker_config.diff_pattern
            )
            if not diff_blocks:
                _record_iteration_runtime(False, stage="diff_parse")
                return process_parallel.SerializableResult(
                    error="No valid diffs found in response", iteration=iteration
                )

            child_code = apply_diff(
                parent.code,
                llm_response,
                process_parallel._worker_config.diff_pattern,
            )
            changes_summary = format_diff_summary(diff_blocks)
        else:
            from openevolve.utils.code_utils import parse_full_rewrite

            new_code = parse_full_rewrite(
                llm_response, process_parallel._worker_config.language
            )

            if not new_code:
                _record_iteration_runtime(False, stage="rewrite_parse")
                return process_parallel.SerializableResult(
                    error="No valid code found in response", iteration=iteration
                )

            child_code = new_code
            changes_summary = "Full rewrite"
        code_phase_sec = time.perf_counter() - t_code_phase

        if len(child_code) > process_parallel._worker_config.max_code_length:
            _record_iteration_runtime(False, stage="length_check")
            return process_parallel.SerializableResult(
                error=(
                    "Generated code exceeds maximum length "
                    f"({len(child_code)} > {process_parallel._worker_config.max_code_length})"
                ),
                iteration=iteration,
            )

        import uuid

        child_id = str(uuid.uuid4())
        t_eval = time.perf_counter()
        try:
            child_metrics = asyncio.run(
                process_parallel._worker_evaluator.evaluate_program(child_code, child_id)
            )
            eval_success = True
        except Exception as e:
            eval_phase_sec = time.perf_counter() - t_eval
            _record_iteration_runtime(False, stage="evaluation", error=str(e))
            return process_parallel.SerializableResult(
                error=f"Evaluation failed: {str(e)}", iteration=iteration
            )
        eval_phase_sec = time.perf_counter() - t_eval

        artifacts = process_parallel._worker_evaluator.get_pending_artifacts(child_id)

        child_program = Program(
            id=child_id,
            code=child_code,
            language=process_parallel._worker_config.language,
            parent_id=parent.id,
            generation=parent.generation + 1,
            metrics=child_metrics,
            iteration_found=iteration,
            metadata={
                "changes": changes_summary,
                "parent_metrics": parent.metrics,
                "island": parent_island,
            },
        )

        iteration_time = time.time() - iteration_start
        _record_iteration_runtime(
            True,
            combined_score=child_metrics.get("combined_score") if isinstance(child_metrics, dict) else None,
            child_metrics=child_metrics if isinstance(child_metrics, dict) else None,
        )

        return process_parallel.SerializableResult(
            child_program_dict=child_program.to_dict(),
            parent_id=parent.id,
            iteration_time=iteration_time,
            prompt=prompt,
            llm_response=llm_response,
            artifacts=artifacts,
            iteration=iteration,
        )
    except Exception as e:
        logger.exception(f"Unexpected error in worker iteration: {e}")
        _record_iteration_runtime(False, stage="worker_exception", error=str(e))
        return process_parallel.SerializableResult(
            error=f"Worker iteration error: {str(e)}", iteration=iteration
        )


def _install_rate_limit(rpm: float) -> None:
    if rpm <= 0:
        return

    min_interval = 60.0 / float(rpm)
    lock = threading.Lock()
    last_by_key: dict[tuple[str, str], float] = {}

    original = OpenAILLM.generate_with_context

    async def wrapped(self, system_message, messages, **kwargs):
        key = (self.api_base, self.model)
        with lock:
            now = time.monotonic()
            last = last_by_key.get(key, 0.0)
            wait_for = min_interval - (now - last)
            if wait_for > 0:
                last_by_key[key] = now + wait_for
            else:
                last_by_key[key] = now
        if wait_for > 0:
            _record_runtime_event(
                "rate_limit_wait",
                {
                    "wait_sec": float(wait_for),
                    "api_base": str(self.api_base),
                    "model": str(self.model),
                },
            )
            await asyncio.sleep(wait_for)
        return await original(self, system_message, messages, **kwargs)

    OpenAILLM.generate_with_context = wrapped


def _extract_code_block(text: str) -> str:
    match = re.search(r"```python\\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def _flatten_message_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(p for p in parts if p)
    return str(content)


def _extract_numeric_usage_fields(obj, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}

    if obj is None:
        return out

    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        key = prefix.rstrip(".")
        if key:
            out[key] = float(obj)
        return out

    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_extract_numeric_usage_fields(v, f"{prefix}{k}."))
        return out

    if isinstance(obj, (list, tuple)):
        for idx, v in enumerate(obj):
            out.update(_extract_numeric_usage_fields(v, f"{prefix}{idx}."))
        return out

    dump = None
    for attr in ("model_dump", "dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                dump = fn()
                break
            except Exception:
                pass
    if isinstance(dump, dict):
        out.update(_extract_numeric_usage_fields(dump, prefix))
        return out

    if hasattr(obj, "__dict__"):
        try:
            out.update(_extract_numeric_usage_fields(vars(obj), prefix))
            return out
        except Exception:
            pass
    return out


def _extract_usage_fields_from_openai_response(response) -> dict[str, float]:
    usage_obj = getattr(response, "usage", None)
    usage_fields = _extract_numeric_usage_fields(usage_obj)
    # Normalize common aliases so plots can use stable names.
    normalized: dict[str, float] = {}
    if "prompt_tokens" in usage_fields:
        normalized["prompt_tokens"] = usage_fields["prompt_tokens"]
    if "completion_tokens" in usage_fields:
        normalized["completion_tokens"] = usage_fields["completion_tokens"]
    if "total_tokens" in usage_fields:
        normalized["total_tokens"] = usage_fields["total_tokens"]
    if "input_tokens" in usage_fields and "prompt_tokens" not in normalized:
        normalized["prompt_tokens"] = usage_fields["input_tokens"]
    if "output_tokens" in usage_fields and "completion_tokens" not in normalized:
        normalized["completion_tokens"] = usage_fields["output_tokens"]
    if (
        "total_tokens" not in normalized
        and "prompt_tokens" in normalized
        and "completion_tokens" in normalized
    ):
        normalized["total_tokens"] = normalized["prompt_tokens"] + normalized["completion_tokens"]
    normalized.update(usage_fields)
    return normalized


def _messages_to_single_prompt(messages: list[dict]) -> str:
    parts: list[str] = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "user")).strip().lower()
        content = _flatten_message_content(msg.get("content", ""))
        if not content:
            continue
        parts.append(f"[{role}]\n{content}")
    return "\n\n".join(parts)


def _openai_model_prefers_completions(model_name: str) -> bool:
    model = (model_name or "").strip().lower()
    if not model:
        return False
    # Optional override list: comma-separated model names.
    raw = os.getenv("OPENEVOLVE_OPENAI_COMPLETIONS_MODELS", "").strip()
    if raw:
        names = [m.strip().lower() for m in raw.split(",") if m.strip()]
        if model in names:
            return True
    return False


def _openai_model_prefers_responses(model_name: str) -> bool:
    model = (model_name or "").strip().lower()
    if not model:
        return False
    # GPT-5 codex variants are served on /v1/responses.
    if model.startswith("gpt-5") and "codex" in model:
        return True
    raw = os.getenv("OPENEVOLVE_OPENAI_RESPONSES_MODELS", "").strip()
    if raw:
        names = [m.strip().lower() for m in raw.split(",") if m.strip()]
        if model in names:
            return True
    return False


def _completion_params_from_chat_params(params: dict) -> dict:
    completion_params = {
        "model": params.get("model"),
        "prompt": _messages_to_single_prompt(params.get("messages") or []),
        "max_tokens": params.get("max_tokens", params.get("max_completion_tokens")),
    }
    if "temperature" in params:
        completion_params["temperature"] = params.get("temperature")
    if "top_p" in params:
        completion_params["top_p"] = params.get("top_p")
    return {k: v for k, v in completion_params.items() if v is not None}


def _responses_params_from_chat_params(params: dict) -> dict:
    response_params = {
        "model": params.get("model"),
        "input": params.get("messages") or [],
        "max_output_tokens": params.get("max_tokens", params.get("max_completion_tokens")),
    }
    if "temperature" in params:
        response_params["temperature"] = params.get("temperature")
    if "top_p" in params:
        response_params["top_p"] = params.get("top_p")
    reasoning_effort = params.get("reasoning_effort")
    if reasoning_effort:
        response_params["reasoning"] = {"effort": reasoning_effort}
    if "service_tier" in params and params.get("service_tier") is not None:
        response_params["service_tier"] = params.get("service_tier")
    return {k: v for k, v in response_params.items() if v is not None}


def _extract_text_from_responses_payload(raw: dict) -> str:
    text = raw.get("output_text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    out_parts: list[str] = []
    for item in raw.get("output") or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content") or []:
            if not isinstance(content, dict):
                continue
            ctext = content.get("text")
            if isinstance(ctext, str) and ctext:
                out_parts.append(ctext)
    return "\n".join(p for p in out_parts if p).strip()


def _call_openai_responses_request(api_base: str, api_key: str, payload: dict) -> tuple[str, dict[str, float]]:
    base = (api_base or "").rstrip("/")
    url = f"{base}/responses"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI responses HTTP {exc.code}: {body}") from exc
    if isinstance(raw, dict) and raw.get("error"):
        raise RuntimeError(f"OpenAI responses error: {raw['error']}")
    text = _extract_text_from_responses_payload(raw or {})
    if not text:
        raise RuntimeError("OpenAI responses returned empty output text")
    usage_fields = _extract_numeric_usage_fields((raw or {}).get("usage") or {})
    normalized: dict[str, float] = {}
    if "input_tokens" in usage_fields:
        normalized["prompt_tokens"] = usage_fields["input_tokens"]
    if "output_tokens" in usage_fields:
        normalized["completion_tokens"] = usage_fields["output_tokens"]
    if "total_tokens" in usage_fields:
        normalized["total_tokens"] = usage_fields["total_tokens"]
    normalized.update(usage_fields)
    return text, normalized


async def _call_openai_responses_and_capture_usage(self, params: dict) -> str:
    api_base = str(getattr(self, "api_base", "") or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"))
    api_key = str(
        getattr(self, "api_key", "") or os.getenv("OPENAI_API_KEY", "")
    ).strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for responses API call")
    payload = _responses_params_from_chat_params(params)
    loop = asyncio.get_event_loop()
    text, usage_fields = await loop.run_in_executor(
        None, lambda: _call_openai_responses_request(api_base, api_key, payload)
    )
    missing = [k for k in ("prompt_tokens", "completion_tokens") if k not in usage_fields]
    if missing:
        _warn_missing_usage_once(
            api_base=api_base,
            model=str(params.get("model", "")),
            source="openai_responses",
            missing_fields=missing,
            available_fields=sorted(usage_fields.keys()),
        )
    setattr(self, "_oe_last_usage_fields", usage_fields)
    return text


async def _call_openai_compatible_and_capture_usage(self, params: dict) -> str:
    loop = asyncio.get_event_loop()
    model_name = str(params.get("model", ""))
    force_responses_first = _openai_model_prefers_responses(model_name)
    force_completions_first = _openai_model_prefers_completions(model_name)
    if force_responses_first:
        _logger.info(
            "Using responses endpoint directly for model %s.", model_name
        )
        return await _call_openai_responses_and_capture_usage(self, params)
    if force_completions_first:
        _logger.info(
            "Using completions endpoint directly for model %s.", model_name
        )
        completion_params = _completion_params_from_chat_params(params)
        response = await loop.run_in_executor(
            None, lambda: self.client.completions.create(**completion_params)
        )
        is_completion_fallback = True
    else:
        try:
            response = await loop.run_in_executor(
                None, lambda: self.client.chat.completions.create(**params)
            )
            is_completion_fallback = False
        except Exception as exc:
            exc_text = str(exc)
            if "v1/responses" in exc_text.lower():
                _logger.warning(
                    "Model %s does not support chat/completions; falling back to responses endpoint.",
                    model_name,
                )
                return await _call_openai_responses_and_capture_usage(self, params)
            elif (
                "not a chat model" in exc_text.lower()
                or "v1/completions" in exc_text.lower()
                or "use v1/completions" in exc_text.lower()
            ):
                _logger.warning(
                    "Model %s does not support chat/completions; falling back to completions endpoint.",
                    model_name,
                )
                completion_params = _completion_params_from_chat_params(params)
                response = await loop.run_in_executor(
                    None, lambda: self.client.completions.create(**completion_params)
                )
                is_completion_fallback = True
            else:
                raise

    usage_fields = _extract_usage_fields_from_openai_response(response)
    missing = [k for k in ("prompt_tokens", "completion_tokens") if k not in usage_fields]
    if missing:
        _warn_missing_usage_once(
            api_base=str(getattr(self, "api_base", "")),
            model=str(getattr(self, "model", "")),
            source="openai_compatible",
            missing_fields=missing,
            available_fields=sorted(usage_fields.keys()),
        )
    setattr(self, "_oe_last_usage_fields", usage_fields)
    if is_completion_fallback:
        text_content = getattr(response.choices[0], "text", "")
        return _flatten_message_content(text_content)
    message_content = response.choices[0].message.content
    return _flatten_message_content(message_content)


def _split_system_and_user_from_messages(messages: list[dict]) -> tuple[str, str]:
    system_parts: list[str] = []
    conversation_parts: list[str] = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "user")).lower()
        content = _flatten_message_content(message.get("content", ""))
        if not content:
            continue
        if role == "system":
            system_parts.append(content)
            continue
        if role == "user":
            conversation_parts.append(content)
            continue
        conversation_parts.append(f"[{role}] {content}")
    return "\n\n".join(system_parts), "\n\n".join(conversation_parts)


def _gemini_native_request(api_key: str, params: dict) -> tuple[str, dict[str, float]]:
    model = str(params.get("model", "")).strip()
    if not model:
        raise RuntimeError("Gemini native call requires a model name")
    model_path = model if model.startswith("models/") else f"models/{model}"

    system_text, user_text = _split_system_and_user_from_messages(params.get("messages") or [])
    if not user_text:
        raise RuntimeError("Gemini native call requires non-empty user content")

    generation_config: dict[str, object] = {}
    max_out = params.get("max_output_tokens", params.get("max_tokens"))
    env_max_out = os.getenv("OPENEVOLVE_GEMINI_MAX_OUTPUT_TOKENS", "").strip()
    if env_max_out:
        try:
            max_out = int(env_max_out)
        except ValueError:
            pass
    if max_out is not None:
        try:
            generation_config["maxOutputTokens"] = int(max_out)
        except (TypeError, ValueError):
            pass
    if "temperature" in params:
        generation_config["temperature"] = params["temperature"]
    if "top_p" in params:
        generation_config["topP"] = params["top_p"]

    payload: dict[str, object] = {
        "contents": [{"role": "user", "parts": [{"text": user_text}]}],
    }
    if system_text:
        payload["systemInstruction"] = {"parts": [{"text": system_text}]}
    if generation_config:
        payload["generationConfig"] = generation_config

    thinking_level_raw = os.getenv("OPENEVOLVE_GEMINI_THINKING_LEVEL", "").strip().lower()
    # Default behavior is auto/provider-default: do not send thinkingConfig.
    thinking_level = ""
    if thinking_level_raw in {"low", "medium", "high"}:
        thinking_level = thinking_level_raw
    elif thinking_level_raw in {"", "auto"}:
        thinking_level = ""
    if thinking_level in {"low", "medium", "high"}:
        generation_config["thinkingConfig"] = {"thinkingLevel": thinking_level}

    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"{model_path}:generateContent?key={urllib.parse.quote(api_key)}"
    )
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    timeout_sec = 300
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini native HTTP {exc.code}: {body}") from exc

    if isinstance(raw, dict) and raw.get("error"):
        raise RuntimeError(f"Gemini native error: {raw['error']}")
    candidates = (raw or {}).get("candidates") or []
    if not candidates:
        raise RuntimeError(f"Gemini native returned no candidates: {raw}")
    first = candidates[0] or {}
    parts = ((first.get("content") or {}).get("parts") or [])
    text = "".join(
        part.get("text", "") for part in parts if isinstance(part, dict) and "text" in part
    ).strip()
    if not text:
        finish_reason = first.get("finishReason", "unknown")
        raise RuntimeError(f"Gemini native returned empty text (finishReason={finish_reason})")
    usage = (raw or {}).get("usageMetadata") or {}
    usage_fields: dict[str, float] = {}
    if isinstance(usage, dict):
        # Normalize Gemini native accounting fields.
        if "promptTokenCount" in usage:
            usage_fields["prompt_tokens"] = float(usage.get("promptTokenCount", 0))
        if "candidatesTokenCount" in usage:
            usage_fields["completion_tokens"] = float(usage.get("candidatesTokenCount", 0))
        if "totalTokenCount" in usage:
            usage_fields["total_tokens"] = float(usage.get("totalTokenCount", 0))
        if "thoughtsTokenCount" in usage:
            usage_fields["thoughts_tokens"] = float(usage.get("thoughtsTokenCount", 0))
        # Keep all raw numeric usage fields too (including nested keys, if present).
        flat_usage = _extract_numeric_usage_fields(usage)
        usage_fields.update({f"gemini_usage.{k}": float(v) for k, v in flat_usage.items()})
    missing = [k for k in ("prompt_tokens", "completion_tokens") if k not in usage_fields]
    if missing:
        _warn_missing_usage_once(
            api_base=_GEMINI_API_PREFIX,
            model=model,
            source="gemini_native",
            missing_fields=missing,
            available_fields=sorted(usage_fields.keys()),
        )
    return text, usage_fields


async def _gemini_native_request_async(
    api_key: str, params: dict
) -> tuple[str, dict[str, float]]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: _gemini_native_request(api_key, params))


def _install_runtime_tracking_overrides() -> None:
    global _RUNTIME_TRACKING_PATCHED
    if _RUNTIME_TRACKING_PATCHED:
        return

    original_generate = OpenAILLM.generate_with_context

    async def wrapped_generate(self, system_message, messages, **kwargs):
        start = time.perf_counter()
        setattr(self, "_oe_last_usage_fields", {})
        prompt_text_parts: list[str] = []
        if isinstance(system_message, str) and system_message:
            prompt_text_parts.append(system_message)
        for msg in messages or []:
            if isinstance(msg, dict):
                prompt_text_parts.append(_flatten_message_content(msg.get("content", "")))
            elif isinstance(msg, str):
                prompt_text_parts.append(msg)
        prompt_text = "\n".join(part for part in prompt_text_parts if part)
        prompt_chars = len(prompt_text)
        prompt_tokens_est = _estimate_tokens_from_text(prompt_text)
        payload_base = {
            "api_base": str(getattr(self, "api_base", "")),
            "model": str(getattr(self, "model", "")),
            "prompt_chars": int(prompt_chars),
            "prompt_tokens_est": int(prompt_tokens_est),
            "prompt_texts": 1 if prompt_chars > 0 else 0,
        }
        try:
            response = await original_generate(self, system_message, messages, **kwargs)
            wall_time = time.perf_counter() - start
            completion_text = response if isinstance(response, str) else str(response)
            completion_chars = len(completion_text)
            completion_tokens_est = _estimate_tokens_from_text(completion_text)
            reported_usage = getattr(self, "_oe_last_usage_fields", {}) or {}
            if not isinstance(reported_usage, dict):
                reported_usage = {}
            _record_runtime_event(
                "llm_call",
                {
                    **payload_base,
                    "success": True,
                    "wall_time_sec": float(wall_time),
                    "completion_chars": int(completion_chars),
                    "completion_tokens_est": int(completion_tokens_est),
                    "completion_texts": 1 if completion_chars > 0 else 0,
                    "reported_usage": reported_usage,
                },
            )
            return response
        except Exception as exc:
            wall_time = time.perf_counter() - start
            reported_usage = getattr(self, "_oe_last_usage_fields", {}) or {}
            if not isinstance(reported_usage, dict):
                reported_usage = {}
            _record_runtime_event(
                "llm_call",
                {
                    **payload_base,
                    "success": False,
                    "wall_time_sec": float(wall_time),
                    "completion_chars": 0,
                    "completion_tokens_est": 0,
                    "completion_texts": 0,
                    "error": str(exc),
                    "reported_usage": reported_usage,
                },
            )
            raise

    OpenAILLM.generate_with_context = wrapped_generate
    _RUNTIME_TRACKING_PATCHED = True


def _install_gemini_overrides() -> None:
    original_generate = OpenAILLM.generate_with_context

    async def wrapped_call(self, params):
        # Anthropic OpenAI-compatible endpoint rejects requests that specify
        # both temperature and top_p for Claude models.
        if str(self.api_base).startswith(_ANTHROPIC_API_PREFIX) or str(self.model).lower().startswith(
            "claude"
        ):
            if "temperature" in params and "top_p" in params:
                params.pop("top_p", None)

        if str(self.api_base).startswith(_GEMINI_API_PREFIX):
            use_native = _env_flag("OPENEVOLVE_GEMINI_NATIVE", default=True)
            if use_native:
                api_key = os.getenv("OPENAI_API_KEY", "").strip()
                if api_key:
                    try:
                        text, usage_fields = await _gemini_native_request_async(api_key, params)
                        setattr(self, "_oe_last_usage_fields", usage_fields)
                        return text
                    except Exception as exc:
                        _logger.warning(
                            "Gemini native call failed; falling back to OpenAI-compatible endpoint: %s",
                            exc,
                        )
        if str(self.api_base).startswith(_OPENAI_API_PREFIX):
            service_tier = os.getenv("OPENAI_SERVICE_TIER", "").strip()
            if service_tier and "service_tier" not in params:
                params["service_tier"] = service_tier
            reasoning_effort = _openai_reasoning_effort_from_env()
            if reasoning_effort and "reasoning_effort" not in params and "reasoning" not in params:
                params["reasoning_effort"] = reasoning_effort
        return await _call_openai_compatible_and_capture_usage(self, params)

    async def wrapped_generate(self, system_message, messages, **kwargs):
        response = await original_generate(self, system_message, messages, **kwargs)
        if not str(self.api_base).startswith(_GEMINI_API_PREFIX):
            return response
        if "<<<<<<< SEARCH" in response and ">>>>>>> REPLACE" in response:
            return response
        return _extract_code_block(response)

    OpenAILLM._call_api = wrapped_call
    OpenAILLM.generate_with_context = wrapped_generate


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    if raw in {"1", "true", "yes", "y"}:
        return True
    if raw in {"0", "false", "no", "n"}:
        return False
    return default


def _openai_reasoning_effort_from_env() -> str:
    raw = os.getenv("OPENAI_REASONING_EFFORT", "").strip().lower()
    if not raw:
        raw = os.getenv("OPENEVOLVE_OPENAI_REASONING_EFFORT", "").strip().lower()
    if raw in {"", "auto", "default", "none"}:
        return ""
    if raw in {"minimal", "low", "medium", "high"}:
        return raw
    _logger.warning(
        "Ignoring unsupported OPENAI_REASONING_EFFORT=%r (expected one of: minimal, low, medium, high, auto)",
        raw,
    )
    return ""


def _select_diverse_programs(
    top_programs: list, num_top: int, num_diverse: int
) -> list:
    if num_diverse <= 0:
        return []
    if len(top_programs) <= num_top:
        return []
    remaining = top_programs[num_top:]
    if not remaining:
        return []
    diverse_count = min(num_diverse, len(remaining))
    if diverse_count <= 0:
        return []
    try:
        # Mirror PromptSampler's random.sample without advancing global state.
        state = random.getstate()
        local_random = random.Random()
        local_random.setstate(state)
        return local_random.sample(remaining, diverse_count)
    except Exception:
        return remaining[:diverse_count]


def _install_prompt_artifact_overrides() -> None:
    global _PROMPT_ARTIFACTS_PATCHED
    if _PROMPT_ARTIFACTS_PATCHED:
        return
    original_build_prompt = PromptSampler.build_prompt
    logger = logging.getLogger(__name__)

    def wrapped_build_prompt(self, *args, **kwargs):
        if getattr(self.config, "include_artifacts", False):
            program_artifacts = kwargs.get("program_artifacts") or {}
            if not isinstance(program_artifacts, dict):
                program_artifacts = {"current_artifacts": program_artifacts}

        top_programs = list(kwargs.get("top_programs") or [])
        if _env_flag("OPENEVOLVE_DISTINCT_SELECTIONS", default=True):
            kwargs["top_programs"] = top_programs
        num_top = min(self.config.num_top_programs, len(top_programs))
        selected_top = top_programs[:num_top]
        diverse = _select_diverse_programs(
            top_programs, num_top, self.config.num_diverse_programs
        )

        inspirations = list(kwargs.get("inspirations") or [])
        desired_inspirations = _get_inspiration_limit(self.config)
        if len(inspirations) > desired_inspirations:
            inspirations = inspirations[:desired_inspirations]
        if _env_flag("OPENEVOLVE_DISTINCT_SELECTIONS", default=True):
            used_ids = {
                p.get("id", "unknown")
                for p in selected_top
            } | {
                p.get("id", "unknown")
                for p in diverse
            }
            inspirations = [
                p
                for p in inspirations
                if p.get("id", "unknown") not in used_ids
            ]
            if desired_inspirations and len(inspirations) < desired_inspirations:
                candidate_pool = list(kwargs.get("previous_programs") or []) + top_programs
                seen_ids = {p.get("id", "unknown") for p in inspirations}
                for program in candidate_pool:
                    program_id = program.get("id", "unknown")
                    if program_id in used_ids or program_id in seen_ids:
                        continue
                    inspirations.append(program)
                    used_ids.add(program_id)
                    seen_ids.add(program_id)
                    if len(inspirations) >= desired_inspirations:
                        break
        kwargs["inspirations"] = inspirations
        logger.info(
            "OE prompt selection top_programs count=%s ids=%s",
            len(selected_top),
            [p.get("id", "unknown") for p in selected_top],
        )
        logger.info(
            "OE prompt selection diverse_programs count=%s ids=%s",
            len(diverse),
            [p.get("id", "unknown") for p in diverse],
        )
        logger.info(
            "OE prompt selection inspiration_programs count=%s ids=%s",
            len(inspirations),
            [p.get("id", "unknown") for p in inspirations],
        )

        prompt = original_build_prompt(self, *args, **kwargs)
        try:
            prompt["oe_selection"] = {
                "top_ids": [p.get("id", "unknown") for p in selected_top],
                "diverse_ids": [p.get("id", "unknown") for p in diverse],
                "inspiration_ids": [p.get("id", "unknown") for p in inspirations],
            }
        except Exception:
            pass

        return prompt

    PromptSampler.build_prompt = wrapped_build_prompt
    _PROMPT_ARTIFACTS_PATCHED = True


def _install_prompt_logging_overrides() -> None:
    global _PROMPT_LOGGING_PATCHED
    if _PROMPT_LOGGING_PATCHED:
        return
    original_log_prompt = ProgramDatabase.log_prompt
    logger = logging.getLogger(__name__)

    def wrapped_log_prompt(self, program_id, template_key, prompt, responses=None):
        try:
            user = prompt.get("user", "")
            selection = prompt.pop("oe_selection", None) or {}
            if selection:
                logger.info(
                    "OE prompt selection program_id=%s top_programs count=%s ids=%s",
                    program_id,
                    len(selection.get("top_ids", [])),
                    selection.get("top_ids", []),
                )
                logger.info(
                    "OE prompt selection program_id=%s diverse_programs count=%s ids=%s",
                    program_id,
                    len(selection.get("diverse_ids", [])),
                    selection.get("diverse_ids", []),
                )
                logger.info(
                    "OE prompt selection program_id=%s inspiration_programs count=%s ids=%s",
                    program_id,
                    len(selection.get("inspiration_ids", [])),
                    selection.get("inspiration_ids", []),
                )
            artifact_blocks = user.count("\nArtifacts:\n```")
            if artifact_blocks:
                logger.info(
                    "Prompt artifacts blocks total=%s",
                    artifact_blocks,
                )
        except Exception:
            pass
        return original_log_prompt(self, program_id, template_key, prompt, responses)

    ProgramDatabase.log_prompt = wrapped_log_prompt
    _PROMPT_LOGGING_PATCHED = True


def _install_process_parallel_overrides() -> None:
    global _PROCESS_PARALLEL_PATCHED
    if _PROCESS_PARALLEL_PATCHED:
        return
    import openevolve.process_parallel as process_parallel

    process_parallel._run_iteration_worker = _patched_run_iteration_worker
    _PROCESS_PARALLEL_PATCHED = True


def _install_database_artifact_overrides() -> None:
    global _DATABASE_ARTIFACTS_PATCHED
    if _DATABASE_ARTIFACTS_PATCHED:
        return
    original_add = ProgramDatabase.add

    def wrapped_add(self, program, iteration=None, target_island=None):
        program_id = program.id
        if (
            program.metadata.get("migrant")
            and not program.artifacts_json
            and not program.artifact_dir
        ):
            parent_id = getattr(program, "parent_id", None)
            if parent_id:
                parent = self.get(parent_id)
                if parent and parent.artifacts_json:
                    program.artifacts_json = parent.artifacts_json
                if parent and parent.artifact_dir:
                    program.artifact_dir = parent.artifact_dir
        result = original_add(self, program, iteration=iteration, target_island=target_island)
        artifacts = _PENDING_ARTIFACTS.pop(program_id, None)
        if artifacts:
            self.store_artifacts(program_id, artifacts)
        return result

    ProgramDatabase.add = wrapped_add
    _DATABASE_ARTIFACTS_PATCHED = True


def _install_evaluator_artifact_storage_overrides() -> None:
    global _EVALUATOR_ARTIFACTS_PATCHED
    if _EVALUATOR_ARTIFACTS_PATCHED:
        return
    original_evaluate = Evaluator.evaluate_program

    async def wrapped_evaluate(self, program_code, program_id=""):
        t0 = time.perf_counter()
        try:
            metrics = await original_evaluate(self, program_code, program_id)
            success = True
            return metrics
        except Exception:
            success = False
            raise
        finally:
            wall_time = time.perf_counter() - t0
            _record_runtime_event(
                "evaluation_call",
                {
                    "success": success,
                    "program_id": str(program_id or ""),
                    "wall_time_sec": float(wall_time),
                },
            )
            if success and program_id and getattr(self, "database", None) is not None:
                artifacts = self.get_pending_artifacts(program_id)
                if artifacts:
                    if self.database.get(program_id):
                        self.database.store_artifacts(program_id, artifacts)
                    else:
                        _PENDING_ARTIFACTS[program_id] = artifacts

    Evaluator.evaluate_program = wrapped_evaluate
    _EVALUATOR_ARTIFACTS_PATCHED = True


def main() -> int:
    # Ensure worker processes load our prompt patches via sitecustomize.
    patch_dir = os.path.join(os.path.dirname(__file__), "_openevolve_patch")
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
    existing_path = os.getenv("PYTHONPATH", "")
    path_parts = existing_path.split(os.pathsep) if existing_path else []
    for entry in (patch_dir, repo_root):
        if entry not in path_parts:
            path_parts.insert(0, entry)
    if path_parts:
        os.environ["PYTHONPATH"] = os.pathsep.join(path_parts)
    os.environ.setdefault("OPENEVOLVE_ENABLE_PATCHES", "1")
    # Make output dir visible to worker processes for runtime tracking.
    _init_runtime_tracking_paths()
    _record_runtime_event(
        "run_start",
        {
            "argv": sys.argv[1:],
            "output_dir": os.getenv("OPENEVOLVE_OUTPUT_DIR", ""),
        },
    )

    _install_gemini_overrides()
    _install_prompt_config_overrides()
    _install_llm_config_overrides()
    _install_evaluator_config_overrides()
    _install_prompt_artifact_overrides()
    _install_prompt_logging_overrides()
    _install_process_parallel_overrides()
    _install_database_artifact_overrides()
    _install_evaluator_artifact_storage_overrides()
    rpm_raw = os.getenv("GEMINI_RPM", "").strip() or os.getenv("OPENEVOLVE_RPM", "").strip()
    if rpm_raw:
        try:
            rpm = float(rpm_raw)
        except ValueError:
            rpm = 0.0
        _install_rate_limit(rpm)
    # Install after provider/rate-limit wrappers so tracked time reflects final behavior.
    _install_runtime_tracking_overrides()
    return openevolve_main()


if __name__ == "__main__":
    raise SystemExit(main())
