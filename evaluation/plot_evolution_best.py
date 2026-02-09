import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


_TS_RE = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(?P<ms>\d{3}) - ")
_INIT_RE = re.compile(r"Evaluated program .*?: (?P<metrics>.+)$")
_ITER_RE = re.compile(r"Iteration (?P<iter>\d+): Program .* completed in")
_METRICS_RE = re.compile(r"Metrics: (?P<metrics>.+)$")


def _parse_metrics(metrics_str: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for part in metrics_str.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        key, raw = part.split("=", 1)
        key = key.strip()
        raw = raw.strip()
        try:
            out[key] = float(raw)
        except ValueError:
            continue
    return out


def _parse_log(path: Path) -> List[Tuple[datetime, int, Dict[str, float]]]:
    events: List[Tuple[datetime, int, Dict[str, float]]] = []
    pending_iter: int | None = None

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        ts_match = _TS_RE.search(line)
        if not ts_match:
            continue
        ts = datetime.strptime(ts_match.group("ts"), "%Y-%m-%d %H:%M:%S")
        ts = ts.replace(microsecond=int(ts_match.group("ms")) * 1000)

        init_match = _INIT_RE.search(line)
        if init_match:
            metrics = _parse_metrics(init_match.group("metrics"))
            if metrics:
                events.append((ts, 0, metrics))
            continue

        iter_match = _ITER_RE.search(line)
        if iter_match:
            pending_iter = int(iter_match.group("iter"))
            continue

        metrics_match = _METRICS_RE.search(line)
        if metrics_match and pending_iter is not None:
            metrics = _parse_metrics(metrics_match.group("metrics"))
            if metrics:
                events.append((ts, pending_iter, metrics))
            pending_iter = None

    return events


def _build_series(
    events: List[Tuple[datetime, int, Dict[str, float]]],
) -> Tuple[List[int], List[Dict[str, float]], List[Dict[str, float]]]:
    events = sorted(events, key=lambda item: item[0])
    steps = list(range(len(events)))
    raw = [event[2] for event in events]

    best: List[Dict[str, float]] = []
    best_score = float("-inf")
    for metrics in raw:
        score = metrics.get("combined_score")
        if score is None:
            best.append(best[-1] if best else metrics)
            continue
        if score >= best_score:
            best_score = score
            best.append(metrics)
        else:
            best.append(best[-1])

    return steps, raw, best


def _expand_group(raw_group: str) -> List[Path]:
    paths: List[Path] = []
    for item in raw_group.split(","):
        item = item.strip()
        if not item:
            continue
        path = Path(item)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.log")))
        else:
            paths.append(path)
    return paths


def _model_bucket(label: str) -> str:
    lower = label.lower()
    if "gpt" in lower:
        return "gpt"
    if "qwen" in lower:
        return "qwen"
    if "gemini" in lower or lower.startswith("gem") or "gem25" in lower:
        return "gemini"
    return "other"


def _collect_series(
    groups: List[List[Path]], labels: List[str]
) -> List[Tuple[str, List[int], List[float], float]]:
    series: List[Tuple[str, List[int], List[float], float]] = []
    for group, label in zip(groups, labels):
        events: List[Tuple[datetime, int, Dict[str, float]]] = []
        for path in group:
            events.extend(_parse_log(path))
        steps, _raw, best = _build_series(events)
        values = [m.get("combined_score", float("nan")) for m in best]
        final = values[-1] if values else float("nan")
        series.append((label, steps, values, final))
    return series


def _adaptive_ylim(values: List[float]) -> Tuple[float, float] | None:
    finite = [v for v in values if v == v]
    if not finite:
        return None
    lo = min(finite)
    hi = max(finite)
    if hi == lo:
        pad = max(1.0, abs(hi) * 0.1)
        return lo - pad, hi + pad
    pad = max((hi - lo) * 0.08, 0.05)
    return lo - pad, hi + pad


def _adaptive_xlim(values: List[int]) -> Tuple[float, float] | None:
    if not values:
        return None
    lo = float(min(values))
    hi = float(max(values))
    if hi == lo:
        return lo - 0.5, hi + 0.5
    pad = max((hi - lo) * 0.03, 0.5)
    return lo - pad, hi + pad


def _plot_progress(
    groups: List[List[Path]], labels: List[str], out_path: Path
) -> Path:
    import matplotlib.pyplot as plt
    from itertools import cycle

    all_series = _collect_series(groups, labels)
    if not all_series:
        raise RuntimeError("No series found to plot.")

    fig, ax_curve = plt.subplots(1, 1, figsize=(12.5, 5.2))

    line_palette = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors) + list(plt.cm.tab20c.colors)
    line_color_cycle = cycle(line_palette)
    all_vals: List[float] = []
    all_steps: List[int] = []
    for label, steps, values, _final in all_series:
        ax_curve.plot(
            steps,
            values,
            linewidth=1.6,
            alpha=0.9,
            label=label,
            color=next(line_color_cycle),
        )
        all_vals.extend(values)
        all_steps.extend(steps)

    ax_curve.set_title("Best combined_score Evolution", fontsize=12)
    ax_curve.set_xlabel("Iteration (time-ordered)", fontsize=10)
    ax_curve.set_ylabel("combined_score", fontsize=10)
    ax_curve.grid(True, linestyle="--", alpha=0.4)
    ax_curve.tick_params(labelsize=9)
    yspan = _adaptive_ylim(all_vals)
    if yspan is not None:
        ax_curve.set_ylim(*yspan)
    xspan = _adaptive_xlim(all_steps)
    if xspan is not None:
        ax_curve.set_xlim(*xspan)

    # Global legend for evolution lines, outside at top.
    line_handles, line_labels = ax_curve.get_legend_handles_labels()
    if line_handles:
        fig.legend(
            line_handles,
            line_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=min(5, max(1, len(line_labels))),
            fontsize=7,
            frameon=True,
            handlelength=2.0,
            columnspacing=1.0,
            labelspacing=0.4,
        )
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_best(groups: List[List[Path]], labels: List[str], out_path: Path) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import numpy as np

    all_series = _collect_series(groups, labels)
    if not all_series:
        raise RuntimeError("No series found to plot.")

    model_colors = {
        "gpt": "#1f77b4",
        "qwen": "#ff7f0e",
        "gemini": "#2ca02c",
        "other": "#7f7f7f",
    }

    valid_series = [s for s in all_series if s[3] == s[3]]
    nan_series = [s for s in all_series if s[3] != s[3]]
    valid_series.sort(key=lambda x: x[3], reverse=True)
    ordered = valid_series + nan_series

    fig, ax_bar = plt.subplots(1, 1, figsize=(11.0, 5.2))
    x = np.arange(len(ordered))
    heights = [final for _label, _steps, _vals, final in ordered]
    labels_order = [label for label, _steps, _vals, _final in ordered]
    bar_colors = [model_colors[_model_bucket(label)] for label in labels_order]
    bars = ax_bar.bar(x, heights, color=bar_colors, alpha=0.9, edgecolor="black", linewidth=0.4)
    for bar, height in zip(bars, heights):
        if height != height:
            continue
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax_bar.set_title("Final combined_score by Config", fontsize=12)
    ax_bar.set_ylabel("combined_score", fontsize=10)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels_order, rotation=80, ha="right", fontsize=8)
    ax_bar.tick_params(axis="y", labelsize=9)
    ax_bar.grid(True, axis="y", linestyle="--", alpha=0.4)
    yspan = _adaptive_ylim(heights)
    if yspan is not None:
        ax_bar.set_ylim(*yspan)

    legend_handles = [
        Patch(facecolor=model_colors["gpt"], edgecolor="black", label="GPT"),
        Patch(facecolor=model_colors["qwen"], edgecolor="black", label="Qwen"),
        Patch(facecolor=model_colors["gemini"], edgecolor="black", label="Gemini"),
        Patch(facecolor=model_colors["other"], edgecolor="black", label="Other"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
        fontsize=8,
        frameon=True,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot best combined_score over time-ordered iterations."
    )
    parser.add_argument(
        "--logs",
        help=(
            "Semicolon-separated groups; each group is a comma-separated list of log paths or "
            "directories containing .log files. Example: logs_a/;logs_b/"
        ),
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-discover logs under openevolve_output*/logs/ in the current directory.",
    )
    parser.add_argument(
        "--labels",
        default="",
        help="Comma-separated labels (optional, must match number of groups).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output file path (e.g. evaluation/plots/best_over_time.pdf).",
    )
    args = parser.parse_args()

    if not args.logs and not args.auto:
        raise SystemExit("--logs or --auto is required.")

    groups: List[List[Path]] = []
    if args.auto:
        for base in sorted(Path("openevolve_output").glob("*/logs")):
            group_paths = sorted(base.glob("*.log"))
            if group_paths:
                groups.append(group_paths)
    else:
        for group in args.logs.split(";"):
            group_paths = _expand_group(group)
            if group_paths:
                groups.append(group_paths)
    if not groups:
        raise SystemExit("No log paths provided.")

    labels = [l.strip() for l in args.labels.split(",") if l.strip()]
    if labels and len(labels) != len(groups):
        raise SystemExit("labels must match number of log groups")
    if not labels:
        auto_labels = []
        for group in groups:
            if not group:
                auto_labels.append("group")
                continue
            base = group[0].parent.parent
            if base.name == "logs" and base.parent.name == "openevolve_output":
                label = base.parent.name
            else:
                label = base.name
            prefix = "openevolve_output_"
            auto_labels.append(label[len(prefix):] if label.startswith(prefix) else label)
        labels = auto_labels

    out_path = Path(args.out)
    if out_path.suffix.lower() != ".pdf":
        out_path = out_path.with_suffix(".pdf")

    progress_out = out_path.with_name(f"{out_path.stem}_progress{out_path.suffix}")
    best_out = out_path.with_name(f"{out_path.stem}_best{out_path.suffix}")

    progress_path = _plot_progress(groups, labels, progress_out)
    print(f"Wrote plot: {progress_path}")
    best_path = _plot_best(groups, labels, best_out)
    print(f"Wrote plot: {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
