#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
DATA_DIR = OUT_DIR / "data"
FIGURES_DIR = OUT_DIR / "figures"

OUTPUT_ROOT = ROOT / "openevolve_output"
ABLATION_ROOT = ROOT / "openevolve_ablation"

# Keep same figure size as plot_gem3flash_seed_diff_full_compare.py
FIGSIZE = (6.6, 4.3)


def _list_version_dirs(root: Path, prefix: str) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for d in sorted(root.glob(f"{prefix}*")):
        if not d.is_dir() or "_v" not in d.name:
            continue
        suffix = d.name.rsplit("_v", 1)[-1]
        if suffix.isdigit():
            out[int(suffix)] = d
    return out


def _read_score_from_info(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        val = payload.get("metrics", {}).get("combined_score")
        return float(val) if val is not None else None
    except Exception:
        return None


def _latest_checkpoint_best_info(run_dir: Path) -> Path | None:
    ckpt_root = run_dir / "checkpoints"
    if not ckpt_root.exists():
        return None
    best_path: Path | None = None
    best_idx = -1
    for d in ckpt_root.glob("checkpoint_*"):
        if not d.is_dir():
            continue
        suffix = d.name.split("checkpoint_", 1)[-1]
        if not suffix.isdigit():
            continue
        idx = int(suffix)
        candidate = d / "best_program_info.json"
        if idx > best_idx and candidate.exists():
            best_idx = idx
            best_path = candidate
    return best_path


def _read_combined_score_with_fallback(run_dir: Path) -> tuple[float | None, str]:
    direct = run_dir / "best" / "best_program_info.json"
    score = _read_score_from_info(direct)
    if score is not None:
        return score, "best"
    fallback = _latest_checkpoint_best_info(run_dir)
    if fallback is not None:
        score = _read_score_from_info(fallback)
        if score is not None:
            return score, "checkpoint_fallback"
    return None, "missing"


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _collect_series_rows(series_specs: list[tuple[str, Path, str]]) -> list[dict]:
    rows: list[dict] = []
    for label, root, prefix in series_specs:
        version_dirs = _list_version_dirs(root, prefix)
        for version, run_dir in sorted(version_dirs.items()):
            score, source = _read_combined_score_with_fallback(run_dir)
            rows.append(
                {
                    "series": label,
                    "version": int(version),
                    "run_dir": str(run_dir),
                    "score_source": source,
                    "combined_score": "" if score is None else float(score),
                    "available": int(score is not None),
                }
            )
    return rows


def _summarize_series(rows: list[dict], order: list[str]) -> list[dict]:
    summary: list[dict] = []
    for series in order:
        vals: list[float] = []
        for r in rows:
            if r["series"] != series:
                continue
            raw = r["combined_score"]
            if isinstance(raw, (int, float)):
                vals.append(float(raw))
            elif str(raw).strip():
                vals.append(float(raw))
        if not vals:
            summary.append(
                {
                    "series": series,
                    "n_runs": 0,
                    "mean": "",
                    "std": "",
                    "min": "",
                    "max": "",
                }
            )
            continue
        summary.append(
            {
                "series": series,
                "n_runs": len(vals),
                "mean": float(statistics.mean(vals)),
                "std": float(statistics.pstdev(vals) if len(vals) > 1 else 0.0),
                "min": float(min(vals)),
                "max": float(max(vals)),
            }
        )
    return summary


def _plot_aggregated_bars(
    summary_rows: list[dict],
    order: list[str],
    colors: dict[str, str],
    out_pdf: Path,
    title: str = "",
) -> None:
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.labelsize": 15,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    means: list[float] = []
    stds: list[float] = []
    counts: list[int] = []
    for name in order:
        rec = next((r for r in summary_rows if r["series"] == name), None)
        if rec is None or str(rec.get("mean", "")).strip() == "":
            means.append(float("nan"))
            stds.append(0.0)
            counts.append(0)
        else:
            means.append(float(rec["mean"]))
            stds.append(float(rec["std"]))
            counts.append(int(rec["n_runs"]))

    x = np.arange(len(order), dtype=float)
    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=5,
        color=[colors.get(k, "#777777") for k in order],
        edgecolor="black",
        linewidth=0.8,
    )

    for bar, m, s, n in zip(bars, means, stds, counts):
        if math.isnan(m):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                0.01,
                "N/A",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        else:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                m + max(s, 0.0) + 0.01,
                f"{m:.3f}",
                ha="center",
                va="bottom",
                fontsize=8.7,
            )

    ax.set_xticks(x, order)
    ax.set_ylabel("Best Combined Score")
    if title:
        ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    fig.savefig(out_pdf)
    plt.close(fig)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Figure 1: Ablation variants aggregated across all available versions.
    ablation_specs = [
        ("111", OUTPUT_ROOT, "gem3flash_pws8_22q_seed_low_full_v"),
        ("011", ABLATION_ROOT, "gem3flash_pws8_22q_noseed_full_v"),
        ("001", ABLATION_ROOT, "gem3flash_pws8_22q_noseed_no_cases_v"),
        ("000", ABLATION_ROOT, "gem3flash_pws8_22q_noseed_no_cases_no_summary_v"),
    ]
    ablation_order = [s[0] for s in ablation_specs]
    ablation_rows = _collect_series_rows(ablation_specs)
    ablation_summary = _summarize_series(ablation_rows, ablation_order)
    _write_csv(DATA_DIR / "gem3flash_ablation_versions_combined_score_raw.csv", ablation_rows)
    _write_csv(DATA_DIR / "gem3flash_ablation_versions_combined_score_summary.csv", ablation_summary)
    _plot_aggregated_bars(
        ablation_summary,
        ablation_order,
        {
            "111": "#4C78A8",
            "011": "#F58518",
            "001": "#54A24B",
            "000": "#B279A2",
        },
        FIGURES_DIR / "gem3flash_ablation_versions_combined_score_aggregated.pdf",
        "",
    )

    # Figure 2: Thinking levels aggregated across all available versions.
    thinking_specs = [
        ("Low (Original)", OUTPUT_ROOT, "gem3flash_pws8_22q_seed_low_full_v"),
        ("Medium", ABLATION_ROOT, "gem3flash_pws8_22q_seed_medium_full_v"),
        ("High", ABLATION_ROOT, "gem3flash_pws8_22q_seed_high_full_v"),
    ]
    thinking_order = [s[0] for s in thinking_specs]
    thinking_rows = _collect_series_rows(thinking_specs)
    thinking_summary = _summarize_series(thinking_rows, thinking_order)
    _write_csv(DATA_DIR / "gem3flash_thinking_versions_combined_score_raw.csv", thinking_rows)
    _write_csv(DATA_DIR / "gem3flash_thinking_versions_combined_score_summary.csv", thinking_summary)
    _plot_aggregated_bars(
        thinking_summary,
        thinking_order,
        {
            "Low (Original)": "#4C78A8",
            "Medium": "#F58518",
            "High": "#54A24B",
        },
        FIGURES_DIR / "gem3flash_thinking_versions_combined_score_aggregated.pdf",
        "",
    )

    print("Wrote figures and CSVs under:", OUT_DIR)


if __name__ == "__main__":
    main()
