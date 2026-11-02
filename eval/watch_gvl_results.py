#!/usr/bin/env python3
"""Continuously render GVL control figures from completed rerun seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score

ROOT = Path("/path/to/counterfactual_study")
DEFAULT_RUN_ROOT = ROOT / "artifacts/rerun_20260922_latest"
SEEDS = (42, 43, 44)
TASKS = (
    "Put the sellotape in the grey box",
    "Put the spoon on the plate",
    "Pour the contents of the cup into the grey box",
    "Remove the lid from pot",
    "Stack up the cups together",
)
TASK_LABELS = (
    "Tape\nin box",
    "Spoon\non plate",
    "Pour\ninto box",
    "Remove\nlid",
    "Stack\ncups",
)
COLORS = {
    "Zero-shot": "#5C677D",
    "Teleop": "#355F8A",
    "Rollout": "#3E7C78",
    "Human": "#B07D3C",
    "Sim HiFi": "#765986",
}
CONDITIONS = {
    "teleop_5shot": "Teleop",
    "rollout_5shot": "Rollout",
    "human_5shot": "Human",
    "hifi_5shot": "Sim HiFi",
}
SCALING_CONDITIONS = {
    "Rollout": {1: "rollout_1shot", 3: "rollout_3shot", 5: "rollout_5shot"},
    "Human": {1: "human_1shot", 3: "human_3shot", 5: "human_5shot"},
    "Sim HiFi": {1: "hifi_1shot", 3: "hifi_3shot", 5: "hifi_5shot"},
}
SCALING_SHOTS = (1, 3, 5)
BOOTSTRAP_SAMPLES = 300
_RECORD_CACHE: dict[Path, list[dict[str, Any]]] = {}
_BOOTSTRAP_CACHE: dict[
    tuple[Path, str | None, int],
    dict[str, list[float]],
] = {}


def _read_metrics(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _payloads(run_root: Path, run_name: str) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for seed in SEEDS:
        payload = _read_metrics(
            run_root / "gvl/runs" / f"{run_name}_seed{seed}" / "paper_metrics.json"
        )
        if payload is not None:
            payloads.append(payload)
    return payloads


def _result_path(run_root: Path, run_name: str, seed: int) -> Path | None:
    candidates = sorted(
        (
            run_root
            / "gvl/runs"
            / f"{run_name}_seed{seed}"
            / "reward_alignment"
        ).glob("*_results.json")
    )
    return candidates[0] if len(candidates) == 1 else None


def _point_metrics(records: list[dict[str, Any]]) -> dict[str, float | None]:
    credit = np.asarray([record["partial_success"] for record in records], dtype=float)
    last = np.asarray([record["progress_pred"][-1] for record in records], dtype=float)
    mean = np.asarray([np.mean(record["progress_pred"]) for record in records], dtype=float)
    maximum = np.asarray([np.max(record["progress_pred"]) for record in records], dtype=float)
    success = credit == 1.0
    levels = np.unique(credit)

    tau_last = kendalltau(credit, last, variant="b").statistic
    tau_mean = kendalltau(credit, mean, variant="b").statistic
    if levels.size >= 2:
        extreme_last = float(last[credit == levels[-1]].mean() - last[credit == levels[0]].mean())
        extreme_mean = float(mean[credit == levels[-1]].mean() - mean[credit == levels[0]].mean())
    else:
        extreme_last = None
        extreme_mean = None
    auc_last = roc_auc_score(success, last) if np.unique(success).size == 2 else None
    auc_mean = roc_auc_score(success, mean) if np.unique(success).size == 2 else None
    return {
        "kendall_tau_last": None if np.isnan(tau_last) else float(tau_last),
        "kendall_tau_mean": None if np.isnan(tau_mean) else float(tau_mean),
        "extreme_reward_difference_last": extreme_last,
        "extreme_reward_difference_mean": extreme_mean,
        "roc_auc_last": None if auc_last is None else float(auc_last),
        "roc_auc_mean": None if auc_mean is None else float(auc_mean),
        "mae_last": float(np.mean(np.abs(last - credit))),
        "mae_mean": float(np.mean(np.abs(mean - credit))),
        "max_progress_minus_partial_credit": float(np.mean(maximum - credit)),
    }


def _bootstrap_task_metrics(
    result_path: Path,
    task: str | None,
) -> dict[str, list[float]]:
    modified_ns = result_path.stat().st_mtime_ns
    cache_key = (result_path, task, modified_ns)
    if cache_key in _BOOTSTRAP_CACHE:
        return _BOOTSTRAP_CACHE[cache_key]
    if result_path not in _RECORD_CACHE:
        _RECORD_CACHE[result_path] = json.loads(result_path.read_text(encoding="utf-8"))
    records = _RECORD_CACHE[result_path]
    if task is not None:
        records = [
            record
            for record in records
            if str(record["task"]) == task
        ]
    values = {
        name: []
        for name in (
            "kendall_tau_last",
            "kendall_tau_mean",
            "extreme_reward_difference_last",
            "extreme_reward_difference_mean",
            "roc_auc_last",
            "roc_auc_mean",
            "mae_last",
            "mae_mean",
            "max_progress_minus_partial_credit",
        )
    }
    stable_seed = int.from_bytes(
        hashlib.sha256(f"{result_path}:{task}".encode()).digest()[:8],
        byteorder="big",
    )
    rng = np.random.default_rng(stable_seed)
    for _ in range(BOOTSTRAP_SAMPLES):
        indices = rng.integers(0, len(records), size=len(records))
        sample_metrics = _point_metrics([records[index] for index in indices])
        for name, value in sample_metrics.items():
            if value is not None:
                values[name].append(value)
    _BOOTSTRAP_CACHE[cache_key] = values
    return values


def _bootstrap_values(
    run_root: Path,
    run_name: str,
    task: str | None,
    metric_name: str,
) -> list[float]:
    values: list[float] = []
    for seed in SEEDS:
        result_path = _result_path(run_root, run_name, seed)
        if result_path is not None:
            values.extend(_bootstrap_task_metrics(result_path, task)[metric_name])
    return values


def _save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.png")
    figure.savefig(temporary, dpi=180, bbox_inches="tight", facecolor="white")
    temporary.replace(path)
    plt.close(figure)


def _style_axis(axis: plt.Axes, title: str, limits: tuple[float, float]) -> None:
    axis.set_title(title, fontsize=10, fontweight="bold")
    axis.set_ylim(*limits)
    axis.grid(axis="y", linestyle=":", color="#D4D7DC", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.axhline(0, color="#777777", linewidth=0.6)


def _bootstrap_violin(
    axis: plt.Axes,
    x: float,
    bootstrap_values: list[float],
    seed_values: list[float],
    color: str,
) -> None:
    if len(bootstrap_values) >= 2 and len(set(bootstrap_values)) > 1:
        violin = axis.violinplot(
            bootstrap_values,
            positions=[x],
            widths=0.13,
            showextrema=False,
            bw_method=0.22,
            points=80,
        )
        body = violin["bodies"][0]
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.42)
        axis.scatter(
            x,
            float(np.median(bootstrap_values)),
            s=12,
            color="white",
            edgecolor=color,
            linewidth=0.8,
            zorder=6,
        )
    if seed_values:
        jitter = np.linspace(-0.025, 0.025, len(seed_values))
        axis.scatter(
            x + jitter,
            seed_values,
            s=18,
            color=color,
            edgecolor="#202428",
            linewidth=0.4,
            zorder=7,
        )


def _control_figure(
    run_root: Path,
    *,
    filename: str,
    metric_specs: tuple[tuple[str, str, tuple[float, float]], ...],
) -> None:
    zero_payloads = _payloads(run_root, "zero_shot")
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.2))
    offsets = np.linspace(-0.27, 0.27, len(CONDITIONS))

    for (metric_name, title, limits), axis in zip(metric_specs, axes, strict=True):
        for boundary in np.arange(0.5, len(TASKS) - 0.5):
            axis.axvline(
                boundary,
                color="#B8BCC2",
                linestyle=(0, (3, 3)),
                linewidth=0.8,
            )
        for task_index, task in enumerate(TASKS):
            zero_values = [
                float(value)
                for payload in zero_payloads
                if (
                    value := payload["by_group"]["task"].get(task, {}).get(metric_name)
                )
                is not None
            ]
            if zero_values:
                axis.hlines(
                    float(np.mean(zero_values)),
                    task_index - 0.42,
                    task_index + 0.42,
                    color=COLORS["Zero-shot"],
                    linestyle=(0, (5, 3)),
                    linewidth=1.5,
                )

        for source_index, (condition, source) in enumerate(CONDITIONS.items()):
            payloads = _payloads(run_root, condition)
            for task_index, task in enumerate(TASKS):
                seed_values = [
                    float(value)
                    for payload in payloads
                    if (
                        value := payload["by_group"]["task"].get(task, {}).get(metric_name)
                    )
                    is not None
                ]
                bootstrap_values = _bootstrap_values(
                    run_root,
                    condition,
                    task,
                    metric_name,
                )
                if seed_values:
                    _bootstrap_violin(
                        axis,
                        task_index + offsets[source_index],
                        bootstrap_values,
                        seed_values,
                        COLORS[source],
                    )

        _style_axis(axis, title, limits)
        axis.set_xticks(range(len(TASKS)), TASK_LABELS, fontsize=8)
        axis.set_xlabel("Benchmark task")

    axes[0].set_ylabel("Metric value")
    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=COLORS["Zero-shot"],
                linestyle=(0, (5, 3)),
                label="Zero-shot",
            ),
            *[
                Line2D([0], [0], color=COLORS[source], marker="D", label=source)
                for source in CONDITIONS.values()
            ],
        ],
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 1.03),
    )
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    _save(figure, run_root / "figures/live/gvl" / filename)


def _scaling_figure(run_root: Path) -> None:
    metric_specs = (
        ("kendall_tau_mean", r"$\tau_{\mathrm{mean}}$", (-0.2, 1.0)),
        ("extreme_reward_difference_mean", r"$\Delta$", (-0.1, 1.0)),
        ("roc_auc_mean", "ROC-AUC", (0.0, 1.0)),
    )
    source_width = 15.5
    latex_scale = 5.5 / source_width
    source_fontsize = 7.5 / latex_scale
    source_tick_fontsize = 6.5 / latex_scale
    offsets = dict(
        zip(SCALING_CONDITIONS, np.linspace(-0.18, 0.18, 3), strict=True)
    )
    zero_payloads = _payloads(run_root, "zero_shot")
    figure, axes = plt.subplots(1, 3, figsize=(source_width, 4.8))
    figure.subplots_adjust(
        left=0.07,
        right=0.99,
        bottom=0.20,
        top=0.76,
        wspace=0.18,
    )

    for (metric_name, title, limits), axis in zip(metric_specs, axes, strict=True):
        zero_values = [
            float(payload["metrics"][metric_name]["value"])
            for payload in zero_payloads
            if payload["metrics"][metric_name]["value"] is not None
        ]
        if zero_values:
            axis.axhline(
                float(np.mean(zero_values)),
                color=COLORS["Zero-shot"],
                linestyle=(0, (5, 3)),
                linewidth=1.5,
            )

        for source, shot_conditions in SCALING_CONDITIONS.items():
            x_values: list[float] = []
            centers: list[float] = []
            for shot, condition in shot_conditions.items():
                payloads = _payloads(run_root, condition)
                seed_values = [
                    float(payload["metrics"][metric_name]["value"])
                    for payload in payloads
                    if payload["metrics"][metric_name]["value"] is not None
                ]
                if not seed_values:
                    continue
                bootstrap_values = _bootstrap_values(
                    run_root,
                    condition,
                    None,
                    metric_name,
                )
                x = SCALING_SHOTS.index(shot) + offsets[source]
                _bootstrap_violin(
                    axis,
                    x,
                    bootstrap_values,
                    seed_values,
                    COLORS[source],
                )
                x_values.append(x)
                centers.append(float(np.mean(seed_values)))
            if len(x_values) > 1:
                order = np.argsort(x_values)
                axis.plot(
                    np.asarray(x_values)[order],
                    np.asarray(centers)[order],
                    color=COLORS[source],
                    linewidth=1.6,
                    zorder=3,
                )

        axis.set_title(
            title,
            fontsize=source_fontsize,
            fontweight="normal",
            pad=20,
        )
        axis.set_ylim(*limits)
        axis.grid(axis="y", linestyle=":", color="#D4D7DC", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.axhline(0, color="#777777", linewidth=0.6)
        axis.set_xticks(
            range(len(SCALING_SHOTS)),
            [str(value) for value in SCALING_SHOTS],
            fontsize=source_tick_fontsize,
        )
        axis.tick_params(axis="y", labelsize=source_tick_fontsize)
        axis.set_xlabel(
            "In-context demonstrations per task",
            fontsize=source_fontsize,
        )
    axes[0].set_ylabel("GVL", fontsize=source_fontsize)

    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=COLORS["Zero-shot"],
                linestyle=(0, (5, 3)),
                label="Zero-shot",
            ),
            *[
                Line2D(
                    [0],
                    [0],
                    color=COLORS[source],
                    marker="D",
                    label=source,
                )
                for source in SCALING_CONDITIONS
            ],
        ],
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.53, 0.99),
        fontsize=source_fontsize,
    )
    output = run_root / "figures/live/gvl/scaling_mean.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.png")
    figure.savefig(temporary, dpi=300, facecolor="white")
    temporary.replace(output)
    figure.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)


def render(run_root: Path) -> None:
    for suffix in ("mean", "last"):
        _control_figure(
            run_root,
            filename=f"control_{suffix}.png",
            metric_specs=(
                (
                    f"kendall_tau_{suffix}",
                    rf"Kendall $\tau_{{\mathrm{{{suffix}}}}}$",
                    (-0.2, 1.0),
                ),
                (
                    f"extreme_reward_difference_{suffix}",
                    f"Extreme {suffix}-progress gap",
                    (-0.1, 1.0),
                ),
                (f"roc_auc_{suffix}", "Success ROC-AUC", (0.0, 1.0)),
            ),
        )

    _control_figure(
        run_root,
        filename="mae_and_max_difference.png",
        metric_specs=(
            ("mae_last", "Partial-credit MAE (last)", (0.0, 0.6)),
            ("mae_mean", "Partial-credit MAE (mean)", (0.0, 0.6)),
            (
                "max_progress_minus_partial_credit",
                "Max progress − partial credit",
                (-0.4, 0.6),
            ),
        ),
    )
    _scaling_figure(run_root)

    status = {
        condition: [
            seed
            for seed in SEEDS
            if (
                run_root
                / "gvl/runs"
                / f"{condition}_seed{seed}"
                / "paper_metrics.json"
            ).is_file()
        ]
        for condition in CONDITIONS
    }
    status.update({
        condition: [
            seed
            for seed in SEEDS
            if (
                run_root
                / "gvl/runs"
                / f"{condition}_seed{seed}"
                / "paper_metrics.json"
            ).is_file()
        ]
        for shot_conditions in SCALING_CONDITIONS.values()
        for condition in shot_conditions.values()
    })
    status["zero_shot"] = [
        seed
        for seed in SEEDS
        if (
            run_root
            / "gvl/runs"
            / f"zero_shot_seed{seed}"
            / "paper_metrics.json"
        ).is_file()
    ]
    status_path = run_root / "figures/live/gvl/status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--interval-seconds", type=float, default=60.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    while True:
        try:
            render(args.run_root.resolve())
            print(
                f"[refreshed] {args.run_root / 'figures/live/gvl'}",
                flush=True,
            )
        except Exception as error:
            print(f"[refresh-failed] {type(error).__name__}: {error}", flush=True)
        if args.once:
            return
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
