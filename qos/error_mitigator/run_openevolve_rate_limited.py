import asyncio
import json
import logging
import os
import random
import re
import threading
import time

from openevolve.cli import main as openevolve_main
from openevolve.llm.openai import OpenAILLM
from openevolve.prompt.sampler import PromptSampler
from openevolve.database import ProgramDatabase
from openevolve.evaluator import Evaluator
from openevolve.utils.metrics_utils import get_fitness_score


_GEMINI_API_PREFIX = "https://generativelanguage.googleapis.com/"
_OPENAI_API_PREFIX = "https://api.openai.com/"
_PROMPT_ARTIFACTS_PATCHED = False
_PROMPT_LOGGING_PATCHED = False
_EVALUATOR_ARTIFACTS_PATCHED = False
_DATABASE_ARTIFACTS_PATCHED = False
_PENDING_ARTIFACTS: dict[str, dict] = {}
_PROMPT_HISTORY_PATCHED = False


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


def _install_gemini_overrides() -> None:
    original_call = OpenAILLM._call_api
    original_generate = OpenAILLM.generate_with_context

    async def wrapped_call(self, params):
        if str(self.api_base).startswith(_GEMINI_API_PREFIX):
            if "max_tokens" in params and "max_output_tokens" not in params:
                params["max_output_tokens"] = params["max_tokens"]
        if str(self.api_base).startswith(_OPENAI_API_PREFIX):
            service_tier = os.getenv("OPENAI_SERVICE_TIER", "").strip()
            if service_tier and "service_tier" not in params:
                params["service_tier"] = service_tier
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


def _stringify_artifact(value) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, indent=2, sort_keys=True)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _load_program_artifacts(program: dict) -> dict:
    artifacts = {}
    artifacts_json = program.get("artifacts_json")
    if artifacts_json:
        try:
            artifacts.update(json.loads(artifacts_json))
        except json.JSONDecodeError:
            artifacts["artifacts_json"] = artifacts_json
    artifact_dir = program.get("artifact_dir")
    if artifact_dir and os.path.isdir(artifact_dir):
        try:
            for name in os.listdir(artifact_dir):
                path = os.path.join(artifact_dir, name)
                if not os.path.isfile(path):
                    continue
                try:
                    with open(path, "rb") as handle:
                        artifacts[name] = handle.read()
                except OSError:
                    continue
        except OSError:
            pass
    return artifacts


def _format_peer_artifacts(programs: list, prefix: str) -> dict:
    payload = {}
    for idx, program in enumerate(programs, start=1):
        artifacts_json = program.get("artifacts_json")
        if artifacts_json:
            content = artifacts_json
        else:
            artifacts = _load_program_artifacts(program)
            if artifacts:
                content = json.dumps(artifacts, sort_keys=True)
            else:
                content = "<no artifacts>"
        payload[f"artifacts_json_{prefix}_{idx}"] = content
    return payload


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


def _format_program_artifacts_block(program: dict, label: str) -> str:
    artifacts_json = program.get("artifacts_json")
    if artifacts_json:
        content = artifacts_json
    else:
        artifacts = _load_program_artifacts(program)
        if artifacts:
            content = json.dumps(artifacts, sort_keys=True)
        else:
            return ""
    return f"Execution Output:\n```\n{content}\n```"


def _install_prompt_history_overrides() -> None:
    global _PROMPT_HISTORY_PATCHED
    if _PROMPT_HISTORY_PATCHED:
        return
    original_history = PromptSampler._format_evolution_history
    original_inspirations = PromptSampler._format_inspirations_section

    def wrapped_history(self, previous_programs, top_programs, inspirations, language, feature_dimensions=None):
        # Largely mirrors upstream behavior, but injects artifacts after key features.
        history_template = self.template_manager.get_template("evolution_history")
        previous_attempt_template = self.template_manager.get_template("previous_attempt")
        top_program_template = self.template_manager.get_template("top_program")
        logger = logging.getLogger(__name__)

        previous_attempts_str = ""
        selected_previous = previous_programs[-min(3, len(previous_programs)) :]

        for i, program in enumerate(reversed(selected_previous)):
            attempt_number = len(previous_programs) - i
            changes = program.get("metadata", {}).get("changes", "Unknown changes")

            performance_parts = []
            for name, value in program.get("metrics", {}).items():
                if isinstance(value, (int, float)):
                    try:
                        performance_parts.append(f"{name}: {value:.4f}")
                    except (ValueError, TypeError):
                        performance_parts.append(f"{name}: {value}")
                else:
                    performance_parts.append(f"{name}: {value}")
            performance_str = ", ".join(performance_parts)

            parent_metrics = program.get("metadata", {}).get("parent_metrics", {})
            outcome = "Mixed results"

            program_metrics = program.get("metrics", {})
            numeric_comparisons_improved = []
            numeric_comparisons_regressed = []

            for m in program_metrics:
                prog_value = program_metrics.get(m, 0)
                parent_value = parent_metrics.get(m, 0)

                if isinstance(prog_value, (int, float)) and isinstance(parent_value, (int, float)):
                    numeric_comparisons_improved.append(prog_value > parent_value)
                    numeric_comparisons_regressed.append(prog_value < parent_value)

            if numeric_comparisons_improved and all(numeric_comparisons_improved):
                outcome = "Improvement in all metrics"
            elif numeric_comparisons_regressed and all(numeric_comparisons_regressed):
                outcome = "Regression in all metrics"

            previous_attempts_str += (
                previous_attempt_template.format(
                    attempt_number=attempt_number,
                    changes=changes,
                    performance=performance_str,
                    outcome=outcome,
                )
                + "\n\n"
            )

        top_programs_str = ""
        selected_top = top_programs[: min(self.config.num_top_programs, len(top_programs))]
        logger.info(
            "OE prompt selection top_programs count=%s ids=%s",
            len(selected_top),
            [p.get("id", "unknown") for p in selected_top],
        )

        for i, program in enumerate(selected_top):
            program_code = program.get("code", "")

            score = get_fitness_score(program.get("metrics", {}), feature_dimensions or [])

            key_features = program.get("key_features", [])
            if not key_features:
                key_features = []
                for name, value in program.get("metrics", {}).items():
                    if isinstance(value, (int, float)):
                        try:
                            key_features.append(f"Performs well on {name} ({value:.4f})")
                        except (ValueError, TypeError):
                            key_features.append(f"Performs well on {name} ({value})")
                    else:
                        key_features.append(f"Performs well on {name} ({value})")

            key_features_str = ", ".join(key_features)
            artifacts_block = _format_program_artifacts_block(program, "Top")
            if artifacts_block:
                key_features_str = f"{key_features_str}\n{artifacts_block}"

            top_programs_str += (
                top_program_template.format(
                    program_number=i + 1,
                    score=f"{score:.4f}",
                    language=language,
                    program_snippet=program_code,
                    key_features=key_features_str,
                )
                + "\n\n"
            )
        if not top_programs_str.strip():
            top_programs_str = "None"

        diverse_programs_str = ""
        diverse_programs = []
        if (
            self.config.num_diverse_programs > 0
            and len(top_programs) > self.config.num_top_programs
        ):
            remaining_programs = top_programs[self.config.num_top_programs :]
            num_diverse = min(self.config.num_diverse_programs, len(remaining_programs))
            if num_diverse > 0:
                diverse_programs = random.sample(remaining_programs, num_diverse)
                diverse_programs_str += "\n\n## Diverse Programs\n\n"
                for i, program in enumerate(diverse_programs):
                    program_code = program.get("code", "")
                    score = get_fitness_score(program.get("metrics", {}), feature_dimensions or [])

                    key_features = program.get("key_features", [])
                    if not key_features:
                        key_features = [
                            f"Alternative approach to {name}"
                            for name in list(program.get("metrics", {}).keys())[:2]
                        ]
                    key_features_str = ", ".join(key_features)
                    artifacts_block = _format_program_artifacts_block(program, "Diverse")
                    if artifacts_block:
                        key_features_str = f"{key_features_str}\n{artifacts_block}"

                    diverse_programs_str += (
                        top_program_template.format(
                            program_number=f"D{i + 1}",
                            score=f"{score:.4f}",
                            language=language,
                            program_snippet=program_code,
                            key_features=key_features_str,
                        )
                        + "\n\n"
                    )
        if not diverse_programs_str.strip():
            diverse_programs_str = "\n\n## Diverse Programs\n\nNone\n\n"

        logger.info(
            "OE prompt selection diverse_programs count=%s ids=%s",
            len(diverse_programs),
            [p.get("id", "unknown") for p in diverse_programs],
        )

        combined_programs_str = top_programs_str + diverse_programs_str

        logger.info(
            "OE prompt selection inspiration_programs count=%s ids=%s",
            len(inspirations),
            [p.get("id", "unknown") for p in inspirations],
        )
        inspirations_section_str = wrapped_inspirations(self, inspirations, language, feature_dimensions)

        return history_template.format(
            previous_attempts=previous_attempts_str.strip(),
            top_programs=combined_programs_str.strip(),
            inspirations_section=inspirations_section_str,
        )

    def wrapped_inspirations(self, inspirations, language, feature_dimensions=None):
        if not inspirations:
            return "## Inspiration Programs\n\nNone"

        inspirations_section_template = self.template_manager.get_template("inspirations_section")
        inspiration_program_template = self.template_manager.get_template("inspiration_program")

        inspiration_programs_str = ""
        for i, program in enumerate(inspirations):
            program_code = program.get("code", "")
            score = get_fitness_score(program.get("metrics", {}), feature_dimensions or [])
            program_type = self._determine_program_type(program, feature_dimensions or [])
            unique_features = self._extract_unique_features(program)
            artifacts_block = _format_program_artifacts_block(program, "Inspiration")
            if artifacts_block:
                unique_features = f"{unique_features}\n{artifacts_block}"

            inspiration_programs_str += (
                inspiration_program_template.format(
                    program_number=i + 1,
                    score=f"{score:.4f}",
                    program_type=program_type,
                    language=language,
                    program_snippet=program_code,
                    unique_features=unique_features,
                )
                + "\n\n"
            )

        return inspirations_section_template.format(
            inspiration_programs=inspiration_programs_str.strip()
        )

    PromptSampler._format_evolution_history = wrapped_history
    PromptSampler._format_inspirations_section = wrapped_inspirations
    _PROMPT_HISTORY_PATCHED = True


def _install_prompt_artifact_overrides() -> None:
    global _PROMPT_ARTIFACTS_PATCHED
    if _PROMPT_ARTIFACTS_PATCHED:
        return
    original_build_prompt = PromptSampler.build_prompt
    logger = logging.getLogger(__name__)

    def wrapped_build_prompt(self, *args, **kwargs):
        include_peer_artifacts = _env_flag("OPENEVOLVE_INCLUDE_PEER_ARTIFACTS", default=False)
        if include_peer_artifacts and getattr(self.config, "include_artifacts", False):
            program_artifacts = kwargs.get("program_artifacts") or {}
            if not isinstance(program_artifacts, dict):
                program_artifacts = {"current_artifacts": program_artifacts}

            top_programs = list(kwargs.get("top_programs") or [])
            if _env_flag("OPENEVOLVE_DISTINCT_SELECTIONS", default=False):
                kwargs["top_programs"] = top_programs
            num_top = min(self.config.num_top_programs, len(top_programs))
            selected_top = top_programs[:num_top]
            diverse = _select_diverse_programs(
                top_programs, num_top, self.config.num_diverse_programs
            )

            inspirations = list(kwargs.get("inspirations") or [])
            if _env_flag("OPENEVOLVE_DISTINCT_SELECTIONS", default=False) and inspirations:
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

        if include_peer_artifacts:
            system_len = len(prompt.get("system", ""))
            user_len = len(prompt.get("user", ""))
            logger.info(
                "Prompt length system=%s user=%s total=%s",
                system_len,
                user_len,
                system_len + user_len,
            )
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


def _install_database_artifact_overrides() -> None:
    global _DATABASE_ARTIFACTS_PATCHED
    if _DATABASE_ARTIFACTS_PATCHED:
        return
    original_add = ProgramDatabase.add

    def wrapped_add(self, program, iteration=None, target_island=None):
        program_id = program.id
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
    _install_prompt_artifact_overrides()
    _install_prompt_history_overrides()
    _install_prompt_logging_overrides()
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
