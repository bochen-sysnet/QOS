import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def _best_by_model(root: Path) -> Dict[str, Tuple[float, Path]]:
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
            best[model] = (float(score), info_path)
    return best


def _load_ratios(info_path: Path) -> Dict[str, float]:
    data = json.loads(info_path.read_text())
    metrics = data.get("metrics") or {}
    return {
        "depth": float(metrics.get("qose_depth", 1.0)),
        "cnot": float(metrics.get("qose_cnot", 1.0)),
        "time": float(metrics.get("avg_run_time", 1.0)),
    }


def main():
    best = _best_by_model(ROOT / "openevolve_output")

    rows = {
        "qos": {"depth": 1.0, "cnot": 1.0, "time": 1.0},
    }
    for model in ("qwen", "gemini", "gpt"):
        info = best.get(model)
        if not info:
            continue
        rows[model] = _load_ratios(info[1])

    labels = list(rows.keys())
    metrics = ["depth", "cnot", "time"]
    x = list(range(len(labels)))
    width = 0.22

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for idx, metric in enumerate(metrics):
        offset = (idx - 1) * width
        vals = [rows[label][metric] for label in labels]
        ax.bar([v + offset for v in x], vals, width, label=metric)

    ax.set_title("Raw ratios (12q runs): QOS vs best QOSE models")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("Ratio (QOSE / QOS)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()

    out_path = Path(__file__).resolve().parent / "plots/qose_ratio_bars.pdf"
    fig.savefig(out_path)
    print(f"Wrote plot: {out_path}")


if __name__ == "__main__":
    main()
