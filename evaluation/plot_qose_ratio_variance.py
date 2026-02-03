import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qos.error_mitigator import evaluator


def _best_by_model(root: Path) -> Dict[str, Path]:
    best: Dict[str, Tuple[float, Path]] = {}
    for info_path in root.rglob("best_program_info.json"):
        try:
            data = json.loads(info_path.read_text())
        except Exception:
            continue
        metrics = data.get("metrics") or {}
        score = metrics.get("combined_score")
        if score is None:
            continue
        lower = str(info_path).lower()
        model = None
        if "qwen" in lower:
            model = "qwen"
        elif "gem" in lower or "gemini" in lower:
            model = "gemini"
        elif "gpt" in lower:
            model = "gpt"
        if model is None:
            continue
        current = best.get(model)
        if current is None or score > current[0]:
            program_path = info_path.parent / "best_program.py"
            if program_path.exists():
                best[model] = (float(score), program_path)
    return {m: p for m, (s, p) in best.items()}


def _run_eval_many(program_path: Path, runs: int) -> List[Dict[str, float]]:
    results = []
    for _ in range(runs):
        metrics, _artifacts = evaluator._evaluate_impl(str(program_path))
        results.append(metrics)
    return results


def main():
    os.environ["QOSE_NUM_SAMPLES"] = "10"
    os.environ["QOSE_SIZE_MIN"] = "12"
    os.environ["QOSE_SIZE_MAX"] = "24"
    os.environ.pop("QOSE_BENCHES", None)

    program_paths: Dict[str, Path] = {
        "qos": ROOT / "qos" / "error_mitigator" / "evolution_target.py",
    }
    program_paths.update(_best_by_model(ROOT / "openevolve_output"))

    methods = [m for m in ("qos", "qwen", "gemini", "gpt") if m in program_paths]
    metrics = ["qose_depth", "qose_cnot", "avg_run_time"]

    means: Dict[str, Dict[str, float]] = {}
    stds: Dict[str, Dict[str, float]] = {}

    for method in methods:
        runs = _run_eval_many(program_paths[method], runs=10)
        means[method] = {}
        stds[method] = {}
        for key in metrics:
            vals = [float(r.get(key, float("nan"))) for r in runs]
            means[method][key] = float(np.nanmean(vals))
            stds[method][key] = float(np.nanstd(vals))

    x = np.arange(len(methods))
    width = 0.22
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    for idx, key in enumerate(metrics):
        vals = [means[m][key] for m in methods]
        errs = [stds[m][key] for m in methods]
        ax.bar(x + (idx - 1) * width, vals, width, yerr=errs, capsize=3, label=key)

    ax.set_title("Average ratios over 10 eval runs (10 random circuits each)")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Ratio (QOSE / QOS)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()

    out_path = Path(__file__).resolve().parent / "qose_ratio_variance.pdf"
    fig.savefig(out_path)
    print(f"Wrote plot: {out_path}")


if __name__ == "__main__":
    main()
