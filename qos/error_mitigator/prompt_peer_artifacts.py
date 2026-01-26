import json
import logging
import os
from typing import Any, Dict, List

from openevolve.prompt.sampler import PromptSampler


def _truthy_env(name: str) -> bool:
    raw = os.getenv(name, "").strip().lower()
    return raw in {"1", "true", "yes", "y"}


def _stringify_artifact(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, indent=2, sort_keys=True)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _load_program_artifacts(program: Dict[str, Any]) -> Dict[str, Any]:
    artifacts: Dict[str, Any] = {}
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


def _format_peer_artifacts(programs: List[Dict[str, Any]], prefix: str) -> Dict[str, str]:
    payload: Dict[str, str] = {}
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


def install_peer_artifact_prompt() -> None:
    if getattr(PromptSampler, "_peer_artifacts_installed", False):
        return

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
            peer_payload: Dict[str, str] = {}
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
    PromptSampler._peer_artifacts_installed = True
