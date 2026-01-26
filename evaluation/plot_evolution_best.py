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


def _plot(groups: List[List[Path]], labels: List[str], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    metrics_keys = ["combined_score"]

    fig, axes = plt.subplots(
        1,
        len(metrics_keys),
        figsize=(4.6 * len(metrics_keys), 4.2),
        sharex=False,
    )
    if len(metrics_keys) == 1:
        axes = [axes]

    for group, label in zip(groups, labels):
        events: List[Tuple[datetime, int, Dict[str, float]]] = []
        for path in group:
            events.extend(_parse_log(path))
        steps, _raw, best = _build_series(events)
        for ax, key in zip(axes, metrics_keys):
            values = [m.get(key, float("nan")) for m in best]
            ax.plot(steps, values, marker="o", markersize=3, label=label)
            ax.set_title(key, fontsize=12)
            ax.grid(True, linestyle="--", alpha=0.4)

    for ax in axes:
        ax.tick_params(labelsize=10)
        ax.set_xlabel("Iteration (time-ordered)", fontsize=10)
    axes[0].legend(loc="best", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


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
            if "general" in label_lower:
                general_groups.append(group)
                general_labels.append(label)
            if "overfit" in label_lower:
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
        if overfit_groups:
            overfit_out = out_path.with_name(f"{out_path.stem}_overfit{out_path.suffix}")
            _plot(overfit_groups, overfit_labels, overfit_out)
            print(f"Wrote plot: {overfit_out}")
    else:
        _plot(groups, labels, out_path)
        print(f"Wrote plot: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
