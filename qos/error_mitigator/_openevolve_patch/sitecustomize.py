import os


def _truthy(raw: str) -> bool:
    return raw.strip().lower() in {"1", "true", "yes", "y"}


if _truthy(os.getenv("OPENEVOLVE_ENABLE_PATCHES", "")):
    try:
        from qos.error_mitigator import run_openevolve_rate_limited as _patches

        for fn_name in (
            "_install_gemini_overrides",
            "_install_prompt_artifact_overrides",
            "_install_process_parallel_overrides",
        ):
            fn = getattr(_patches, fn_name, None)
            if callable(fn):
                fn()
    except Exception:
        # Avoid breaking worker startup if patching fails.
        pass
