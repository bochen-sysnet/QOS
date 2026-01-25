import asyncio
import json
import logging
import os
import re
import threading
import time

from openevolve.cli import main as openevolve_main
from openevolve.llm.openai import OpenAILLM
from openevolve.prompt.sampler import PromptSampler


_GEMINI_API_PREFIX = "https://generativelanguage.googleapis.com/"
_OPENAI_API_PREFIX = "https://api.openai.com/"


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


def _truthy_env(name: str) -> bool:
    raw = os.getenv(name, "").strip().lower()
    return raw in {"1", "true", "yes", "y"}


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
        artifacts = _load_program_artifacts(program)
        if not artifacts:
            content = "<no artifacts>"
        else:
            parts = []
            for key in sorted(artifacts.keys()):
                parts.append(f"{key}:\n{_stringify_artifact(artifacts[key])}")
            content = "\n\n".join(parts)
        program_id = program.get("id", "unknown")
        payload[f"{prefix}_{idx}_artifacts (id={program_id})"] = content
    return payload


def _install_prompt_artifact_overrides() -> None:
    original_build_prompt = PromptSampler.build_prompt
    logger = logging.getLogger(__name__)

    def wrapped_build_prompt(self, *args, **kwargs):
        if not _truthy_env("OPENEVOLVE_INCLUDE_PEER_ARTIFACTS"):
            prompt = original_build_prompt(self, *args, **kwargs)
        elif not getattr(self.config, "include_artifacts", False):
            prompt = original_build_prompt(self, *args, **kwargs)
        else:
            program_artifacts = kwargs.get("program_artifacts") or {}
            if not isinstance(program_artifacts, dict):
                program_artifacts = {"current_artifacts": program_artifacts}

            top_programs = list(kwargs.get("top_programs") or [])
            num_top = min(self.config.num_top_programs, len(top_programs))
            selected_top = top_programs[:num_top]
            peer_payload = {}
            if selected_top:
                peer_payload.update(_format_peer_artifacts(selected_top, "top_program"))

            num_diverse = self.config.num_diverse_programs
            if num_diverse > 0 and len(top_programs) > num_top:
                remaining = top_programs[num_top:]
                diverse_count = min(num_diverse, len(remaining))
                diverse = remaining[:diverse_count]
                if diverse:
                    peer_payload.update(_format_peer_artifacts(diverse, "diverse_program"))

            program_artifacts = {**program_artifacts, **peer_payload}
            kwargs["program_artifacts"] = program_artifacts
            prompt = original_build_prompt(self, *args, **kwargs)

        if _truthy_env("OPENEVOLVE_INCLUDE_PEER_ARTIFACTS"):
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


def main() -> int:
    _install_gemini_overrides()
    _install_prompt_artifact_overrides()
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
