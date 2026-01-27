import os


def _truthy(raw: str) -> bool:
    return raw.strip().lower() in {"1", "true", "yes", "y"}


if _truthy(os.getenv("OPENEVOLVE_ENABLE_PATCHES", "")):
    try:
        from qos.error_mitigator import run_openevolve_rate_limited as _patches

        _patches._install_prompt_artifact_overrides()
        _patches._install_prompt_history_overrides()
    except Exception:
        # Avoid breaking worker startup if patching fails.
        pass
