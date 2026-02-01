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


def _plot(groups: List[List[Path]], labels: List[str], out_path: Path) -> Path:
    import matplotlib.pyplot as plt
    import textwrap
    from cycler import cycler

    metrics_keys = ["combined_score"]
    display_labels = labels

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12.0, 4.5),
        sharex=False,
        sharey=True,
    )
    line_axes = list(axes)

    axis_map = {
        "gpt": line_axes[0],
        "qwen": line_axes[1],
        "gemini": line_axes[2],
    }

    colors = list(plt.cm.tab10.colors)
    from itertools import cycle
    color_cycle = cycle(colors)
    plotted_axes = {ax: False for ax in line_axes}

    def _model_bucket(label: str) -> str | None:
        lower = label.lower()
        if "gpt" in lower:
            return "gpt"
        if "qwen" in lower:
            return "qwen"
        if "gemini" in lower or lower.startswith("gem") or " gem" in lower or "_gem" in lower:
            return "gemini"
        return None

    def _linestyle_for_label(label: str) -> str:
        return "-"

    final_scores: dict[str, float] = {}
    min_score = -1000.0

    for group, label in zip(groups, display_labels):
        events: List[Tuple[datetime, int, Dict[str, float]]] = []
        for path in group:
            events.extend(_parse_log(path))
        steps, _raw, best = _build_series(events)
        bucket = _model_bucket(label)
        target_axes = [axis_map[bucket]] if bucket else line_axes
        linestyle = _linestyle_for_label(label)
        for ax, key in zip(target_axes, metrics_keys):
            values = [m.get(key, float("nan")) for m in best]
            values = [v if v == v and v > min_score else float("nan") for v in values]
            ax.plot(
                steps,
                values,
                linewidth=1.6,
                alpha=0.9,
                label=label,
                color=next(color_cycle),
                linestyle=linestyle,
            )
            ax.set_title(key, fontsize=12)
            ax.grid(True, linestyle="--", alpha=0.4)
            plotted_axes[ax] = True
        if best:
            final_scores[label] = best[-1].get("combined_score", float("nan"))

    for ax in line_axes:
        ax.tick_params(labelsize=10)
        ax.set_xlabel("Iteration (time-ordered)", fontsize=10)
        if plotted_axes[ax]:
            ax.legend(
                loc="lower center",
                bbox_to_anchor=(0.5, 0.03),
                borderaxespad=0.0,
                fontsize=7,
                frameon=True,
                ncol=1,
                columnspacing=1.2,
                labelspacing=0.7,
                handletextpad=0.5,
                handlelength=2.0,
            )
        else:
            ax.text(
                0.5,
                0.5,
                "No runs",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                color="gray",
            )

    line_axes[0].set_title("combined_score (gpt)", fontsize=11)
    line_axes[1].set_title("combined_score (qwen)", fontsize=11)
    line_axes[2].set_title("combined_score (gemini)", fontsize=11)

    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _plot_final_scores(out_path: Path, labels: List[str], final_scores: dict[str, float]) -> Path:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.5), sharey=True)
    axis_map = {
        "gpt": axes[0],
        "qwen": axes[1],
        "gemini": axes[2],
    }

    def _model_bucket(label: str) -> str | None:
        lower = label.lower()
        if "gpt" in lower:
            return "gpt"
        if "qwen" in lower:
            return "qwen"
        if "gemini" in lower or lower.startswith("gem") or " gem" in lower or "_gem" in lower:
            return "gemini"
        return None

    for model, ax in axis_map.items():
        labels_order = [l for l in final_scores.keys() if _model_bucket(l) == model]
        x = list(range(len(labels_order)))
        heights = [final_scores[label] for label in labels_order]
        bars = ax.bar(x, heights, width=0.7, alpha=0.9)
        for bar in bars:
            height = bar.get_height()
            if height != height:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

        ax.set_title(f"Final combined_score ({model})", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_order, rotation=90, ha="center", fontsize=8)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        if not labels_order:
            ax.text(
                0.5,
                0.5,
                "No runs",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                color="gray",
            )

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _collect_final_scores(
    groups: List[List[Path]], labels: List[str]
) -> dict[str, float]:
    final_scores: dict[str, float] = {}
    for group, label in zip(groups, labels):
        events: List[Tuple[datetime, int, Dict[str, float]]] = []
        for path in group:
            events.extend(_parse_log(path))
        _steps, _raw, best = _build_series(events)
        if best:
            final_scores[label] = best[-1].get("combined_score", float("nan"))
    return final_scores


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
        for base in sorted(Path(".").glob("openevolve_output*/logs")):
            group_paths = sorted(base.glob("*.log"))
            if group_paths:
                groups.append(group_paths)
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

    def _split_groups(
        groups: List[List[Path]], labels: List[str]
    ) -> tuple[list[list[Path]], list[str], list[list[Path]], list[str]]:
        general_groups: list[list[Path]] = []
        general_labels: list[str] = []
        overfit_groups: list[list[Path]] = []
        overfit_labels: list[str] = []
        for group, label in zip(groups, labels):
            label_lower = label.lower()
            if (
                "general" in label_lower
                or "_gen_" in label_lower
                or label_lower.startswith("gen_")
                or label_lower.endswith("_gen")
            ):
                general_groups.append(group)
                general_labels.append(label)
            if (
                "overfit" in label_lower
                or "_of_" in label_lower
                or label_lower.startswith("of_")
                or label_lower.endswith("_of")
            ):
                overfit_groups.append(group)
                overfit_labels.append(label)
        return general_groups, general_labels, overfit_groups, overfit_labels

    out_path = Path(args.out)
    if out_path.suffix.lower() != ".pdf":
        out_path = out_path.with_suffix(".pdf")

    general_groups, general_labels, overfit_groups, overfit_labels = _split_groups(
        groups, labels
    )
    if general_groups or overfit_groups:
        if general_groups:
            general_out = out_path.with_name(f"{out_path.stem}_general{out_path.suffix}")
            _plot(general_groups, general_labels, general_out)
            print(f"Wrote plot: {general_out}")
            general_final_out = out_path.with_name(
                f"{out_path.stem}_general_final_scores{out_path.suffix}"
            )
            general_final = _collect_final_scores(general_groups, general_labels)
            final_path = _plot_final_scores(general_final_out, general_labels, general_final)
            print(f"Wrote plot: {final_path}")
        if overfit_groups:
            overfit_out = out_path.with_name(f"{out_path.stem}_overfit{out_path.suffix}")
            _plot(overfit_groups, overfit_labels, overfit_out)
            print(f"Wrote plot: {overfit_out}")
            overfit_final_out = out_path.with_name(
                f"{out_path.stem}_overfit_final_scores{out_path.suffix}"
            )
            overfit_final = _collect_final_scores(overfit_groups, overfit_labels)
            final_path = _plot_final_scores(overfit_final_out, overfit_labels, overfit_final)
            print(f"Wrote plot: {final_path}")
    else:
        plot_path = _plot(groups, labels, out_path)
        print(f"Wrote plot: {plot_path}")
        final_out = out_path.with_name(f"{out_path.stem}_final_scores{out_path.suffix}")
        final_scores = _collect_final_scores(groups, labels)
        final_path = _plot_final_scores(final_out, labels, final_scores)
        print(f"Wrote plot: {final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
