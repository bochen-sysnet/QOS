import asyncio
import json
import logging
import os
import random
import re
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
_logger = logging.getLogger(__name__)


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

        iteration_start = time.time()

        try:
            llm_response = asyncio.run(
                process_parallel._worker_llm_ensemble.generate_with_context(
                    system_message=prompt["system"],
                    messages=[{"role": "user", "content": prompt["user"]}],
                )
            )
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return process_parallel.SerializableResult(
                error=f"LLM generation failed: {str(e)}", iteration=iteration
            )

        if llm_response is None:
            return process_parallel.SerializableResult(
                error="LLM returned None response", iteration=iteration
            )

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
                return process_parallel.SerializableResult(
                    error="No valid code found in response", iteration=iteration
                )

            child_code = new_code
            changes_summary = "Full rewrite"

        if len(child_code) > process_parallel._worker_config.max_code_length:
            return process_parallel.SerializableResult(
                error=(
                    "Generated code exceeds maximum length "
                    f"({len(child_code)} > {process_parallel._worker_config.max_code_length})"
                ),
                iteration=iteration,
            )

        import uuid

        child_id = str(uuid.uuid4())
        child_metrics = asyncio.run(
            process_parallel._worker_evaluator.evaluate_program(child_code, child_id)
        )

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


def _gemini_native_request(api_key: str, params: dict) -> str:
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
    return text


async def _gemini_native_request_async(api_key: str, params: dict) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: _gemini_native_request(api_key, params))


def _install_gemini_overrides() -> None:
    original_call = OpenAILLM._call_api
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
                        return await _gemini_native_request_async(api_key, params)
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
        return await original_call(self, params)

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
        metrics = await original_evaluate(self, program_code, program_id)
        if program_id and getattr(self, "database", None) is not None:
            artifacts = self.get_pending_artifacts(program_id)
            if artifacts:
                if self.database.get(program_id):
                    self.database.store_artifacts(program_id, artifacts)
                else:
                    _PENDING_ARTIFACTS[program_id] = artifacts
        return metrics

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
    return openevolve_main()


if __name__ == "__main__":
    raise SystemExit(main())
