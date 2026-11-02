#!/usr/bin/env python3
"""Continuously render figures from whatever rerun results are complete."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score

import watch_gvl_results as gvl_results

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
TASK_LABELS = ("Tape\nin box", "Spoon\non plate", "Pour\ninto box", "Remove\nlid", "Stack\ncups")
DEMO_COUNTS = (5, 10, 20, 50)
COLORS = {
    "Zero-shot": "#5C677D",
    "Teleop": "#355F8A",
    "Rollout": "#3E7C78",
    "Human": "#B07D3C",
    "Sim HiFi": "#765986",
    "Sim LoFi": "#B45F68",
    "Control mix": "#C05A3D",
    "Scaled mix": "#2F7E9E",
}
CONDITIONS = {
    "teleop5": ("Teleop", 5),
    "rollout5": ("Rollout", 5),
    "human5": ("Human", 5),
    "hifi5": ("Sim HiFi", 5),
    "rollout10": ("Rollout", 10),
    "human10": ("Human", 10),
    "human20": ("Human", 20),
    "hifi10": ("Sim HiFi", 10),
    "hifi20": ("Sim HiFi", 20),
    "hifi50": ("Sim HiFi", 50),
    "lofi5": ("Sim LoFi", 5),
    "lofi50": ("Sim LoFi", 50),
}
CONTROL = ("teleop5", "rollout5", "human5", "hifi5")
MIX_CONDITIONS = (
    ("mix_control", "Control mix (5/5/5/5)", "Control mix"),
    ("mix_scaled", "Scaled mix (5/10/20/50)", "Scaled mix"),
)
GVL_CONDITIONS = (
    "zero_shot",
    "teleop_5shot",
    "rollout_5shot",
    "human_5shot",
    "hifi_5shot",
)
BOOTSTRAP_SAMPLES = 300
_RECORD_CACHE: dict[Path, list[dict[str, Any]]] = {}
_BOOTSTRAP_CACHE: dict[
    tuple[Path, str | None, int],
    dict[str, list[float]],
] = {}

plt.rcParams.update(
    {
        # ICLR uses Times at 10 pt. STIXGeneral is the available Times-like
        # font and remains embedded as editable TrueType text in the PDF.
        "font.family": "STIXGeneral",
        "font.size": 10,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def _read_metrics(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.png")
    figure.savefig(temporary, dpi=180, bbox_inches="tight", facecolor="white")
    temporary.replace(path)
    plt.close(figure)


def _style_axis(
    axis: plt.Axes,
    title: str,
    limits: tuple[float, float],
    *,
    title_fontsize: float = 10,
    title_fontweight: str = "bold",
    title_pad: float | None = None,
) -> None:
    axis.set_title(
        title,
        fontsize=title_fontsize,
        fontweight=title_fontweight,
        pad=title_pad,
    )
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
    alpha: float = 0.42,
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
        body.set_alpha(alpha)
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


def _condition_payloads(run_root: Path, condition: str) -> list[dict[str, Any]]:
    payloads = []
    for seed in SEEDS:
        payload = _read_metrics(
            run_root / "eval" / f"{condition}_seed{seed}" / "paper_metrics.json"
        )
        if payload is not None:
            payloads.append(payload)
    return payloads


def _result_path(run_root: Path, run_name: str) -> Path | None:
    candidates = sorted(
        (run_root / "eval" / run_name / "reward_alignment").glob("*_results.json")
    )
    return candidates[0] if len(candidates) == 1 else None


def _point_metrics(records: list[dict[str, Any]]) -> dict[str, float | None]:
    credit = np.asarray([record["partial_success"] for record in records], dtype=float)
    last = np.asarray([record["progress_pred"][-1] for record in records], dtype=float)
    mean = np.asarray([np.mean(record["progress_pred"]) for record in records], dtype=float)
    maximum = np.asarray([np.max(record["progress_pred"]) for record in records], dtype=float)
    tasks = np.asarray([str(record["task"]) for record in records])
    success = credit == 1.0

    tau_last = kendalltau(credit, last, variant="b").statistic
    tau_mean = kendalltau(credit, mean, variant="b").statistic
    extreme_gaps: list[tuple[float, float]] = []
    for task in np.unique(tasks):
        mask = tasks == task
        task_credit = credit[mask]
        levels = np.unique(task_credit)
        if levels.size < 2:
            continue
        task_last = last[mask]
        task_mean = mean[mask]
        extreme_gaps.append(
            (
                float(
                    task_last[task_credit == levels[-1]].mean()
                    - task_last[task_credit == levels[0]].mean()
                ),
                float(
                    task_mean[task_credit == levels[-1]].mean()
                    - task_mean[task_credit == levels[0]].mean()
                ),
            )
        )
    extreme_last = (
        float(np.mean([gap[0] for gap in extreme_gaps])) if extreme_gaps else None
    )
    extreme_mean = (
        float(np.mean([gap[1] for gap in extreme_gaps])) if extreme_gaps else None
    )
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


def _bootstrap_metrics(
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
        records = [record for record in records if str(record["task"]) == task]
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


def _condition_bootstrap_values(
    run_root: Path,
    condition: str,
    metric_name: str,
    *,
    task: str | None = None,
) -> list[float]:
    values: list[float] = []
    for seed in SEEDS:
        result_path = _result_path(run_root, f"{condition}_seed{seed}")
        if result_path is not None:
            values.extend(_bootstrap_metrics(result_path, task)[metric_name])
    return values


def control_figure(run_root: Path, *, last: bool) -> None:
    suffix = "last" if last else "mean"
    metric_specs = (
        (f"kendall_tau_{suffix}", rf"Kendall $\tau_{{\mathrm{{{suffix}}}}}$", (-0.2, 1.0)),
        (
            f"extreme_reward_difference_{suffix}",
            f"Extreme {suffix}-progress gap",
            (-0.1, 1.0),
        ),
        (f"roc_auc_{suffix}", "Success ROC-AUC", (0.0, 1.0)),
    )
    zero = _read_metrics(run_root / "eval/zero_shot/paper_metrics.json")
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.2))
    offsets = np.linspace(-0.27, 0.27, len(CONTROL))
    for metric_name, title, limits, axis in (
        (*spec, axis) for spec, axis in zip(metric_specs, axes, strict=True)
    ):
        for boundary in np.arange(0.5, len(TASKS) - 0.5):
            axis.axvline(boundary, color="#B8BCC2", linestyle=(0, (3, 3)), linewidth=0.8)
        if zero is not None:
            by_task = zero["by_group"]["task"]
            for task_index, task in enumerate(TASKS):
                value = by_task.get(task, {}).get(metric_name)
                if value is not None:
                    axis.hlines(
                        value,
                        task_index - 0.42,
                        task_index + 0.42,
                        color=COLORS["Zero-shot"],
                        linestyle=(0, (5, 3)),
                        linewidth=1.5,
                    )
        for source_index, condition in enumerate(CONTROL):
            source, _ = CONDITIONS[condition]
            payloads = _condition_payloads(run_root, condition)
            for task_index, task in enumerate(TASKS):
                seed_values = [
                    float(value)
                    for payload in payloads
                    if (
                        value := payload["by_group"]["task"].get(task, {}).get(metric_name)
                    )
                    is not None
                ]
                bootstrap_values = _condition_bootstrap_values(
                    run_root,
                    condition,
                    metric_name,
                    task=task,
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
                Line2D([0], [0], color=COLORS[CONDITIONS[name][0]], marker="D", label=CONDITIONS[name][0])
                for name in CONTROL
            ],
        ],
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 1.03),
    )
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    _save(figure, run_root / "figures/live" / f"control_{suffix}.png")


def combined_control_mean_figure(run_root: Path) -> None:
    """Keep the wide layout while compensating fonts for 5.5-inch inclusion."""
    metric_specs = (
        ("kendall_tau_mean", r"$\tau_{\mathrm{mean}}$", (-0.2, 1.0)),
        (
            "extreme_reward_difference_mean",
            r"$\Delta$",
            (-0.1, 1.0),
        ),
        ("roc_auc_mean", "ROC-AUC", (0.0, 1.0)),
    )
    source_width = 15.5
    latex_width = 5.5
    latex_scale = latex_width / source_width
    source_fontsize = 7.5 / latex_scale
    source_tick_fontsize = 6.5 / latex_scale
    compact_task_labels = ("Tape", "Spoon", "Pour", "Lid", "Cups")

    figure, axes = plt.subplots(2, 3, figsize=(source_width, 8.0))
    figure.subplots_adjust(
        left=0.07,
        right=0.99,
        bottom=0.08,
        top=0.82,
        hspace=0.58,
        wspace=0.18,
    )

    robometer_zero = _read_metrics(run_root / "eval/zero_shot/paper_metrics.json")
    gvl_zero = gvl_results._payloads(run_root, "zero_shot")
    offsets = np.linspace(-0.27, 0.27, len(CONTROL))

    for row_index, row_axes in enumerate(axes):
        for metric_index, ((metric_name, title, limits), axis) in enumerate(
            zip(metric_specs, row_axes, strict=True)
        ):
            for boundary in np.arange(0.5, len(TASKS) - 0.5):
                axis.axvline(
                    boundary,
                    color="#B8BCC2",
                    linestyle=(0, (3, 3)),
                    linewidth=0.8,
                )

            for task_index, task in enumerate(TASKS):
                if row_index == 0 and robometer_zero is not None:
                    zero_value = (
                        robometer_zero["by_group"]["task"]
                        .get(task, {})
                        .get(metric_name)
                    )
                    zero_values = [] if zero_value is None else [float(zero_value)]
                else:
                    zero_values = [
                        float(value)
                        for payload in gvl_zero
                        if (
                            value := payload["by_group"]["task"]
                            .get(task, {})
                            .get(metric_name)
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

            for source_index, robometer_condition in enumerate(CONTROL):
                source = CONDITIONS[robometer_condition][0]
                if row_index == 0:
                    payloads = _condition_payloads(run_root, robometer_condition)
                    gvl_condition = ""
                else:
                    gvl_condition = tuple(gvl_results.CONDITIONS)[source_index]
                    payloads = gvl_results._payloads(run_root, gvl_condition)
                for task_index, task in enumerate(TASKS):
                    seed_values = [
                        float(value)
                        for payload in payloads
                        if (
                            value := payload["by_group"]["task"]
                            .get(task, {})
                            .get(metric_name)
                        )
                        is not None
                    ]
                    if row_index == 0:
                        bootstrap_values = _condition_bootstrap_values(
                            run_root,
                            robometer_condition,
                            metric_name,
                            task=task,
                        )
                    else:
                        bootstrap_values = gvl_results._bootstrap_values(
                            run_root,
                            gvl_condition,
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

            _style_axis(
                axis,
                title if row_index == 0 else "",
                limits,
                title_fontsize=source_fontsize,
                title_fontweight="normal",
                title_pad=20,
            )
            axis.set_xticks(
                range(len(TASKS)),
                compact_task_labels,
                fontsize=source_tick_fontsize,
            )
            axis.tick_params(axis="y", labelsize=source_tick_fontsize)
            if metric_index == 0:
                axis.set_ylabel(
                    "Robometer" if row_index == 0 else "GVL",
                    fontsize=source_fontsize,
                )

    legend_handles = [
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
                color=COLORS[CONDITIONS[name][0]],
                marker="D",
                label=CONDITIONS[name][0],
            )
            for name in CONTROL
        ],
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.54, 0.99),
        fontsize=source_fontsize,
    )

    png_path = run_root / "figures/live/control_mean.png"
    png_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = png_path.with_name(f".{png_path.name}.tmp.png")
    figure.savefig(temporary, dpi=300, facecolor="white")
    temporary.replace(png_path)
    figure.savefig(
        png_path.with_suffix(".pdf"),
        facecolor="white",
    )
    plt.close(figure)


def mixed_source_mean_figure(run_root: Path) -> None:
    """Plot the two mixed-source Robometer variants per benchmark task."""
    metric_specs = (
        ("kendall_tau_mean", r"$\tau_{\mathrm{mean}}$", (-0.2, 1.0)),
        ("extreme_reward_difference_mean", r"$\Delta$", (-0.1, 1.0)),
        ("roc_auc_mean", "ROC-AUC", (0.0, 1.0)),
    )
    source_width = 15.5
    latex_scale = 5.5 / source_width
    source_fontsize = 7.5 / latex_scale
    source_tick_fontsize = 6.5 / latex_scale
    compact_task_labels = ("Tape", "Spoon", "Pour", "Lid", "Cups")
    offsets = (-0.13, 0.13)

    figure, axes = plt.subplots(1, 3, figsize=(source_width, 4.8))
    figure.subplots_adjust(
        left=0.07,
        right=0.99,
        bottom=0.20,
        top=0.76,
        wspace=0.18,
    )
    zero = _read_metrics(run_root / "eval/zero_shot/paper_metrics.json")

    for (metric_name, title, limits), axis in zip(metric_specs, axes, strict=True):
        for boundary in np.arange(0.5, len(TASKS) - 0.5):
            axis.axvline(
                boundary,
                color="#B8BCC2",
                linestyle=(0, (3, 3)),
                linewidth=0.8,
            )
        if zero is not None:
            for task_index, task in enumerate(TASKS):
                value = (
                    zero["by_group"]["task"].get(task, {}).get(metric_name)
                )
                if value is not None:
                    axis.hlines(
                        float(value),
                        task_index - 0.42,
                        task_index + 0.42,
                        color=COLORS["Zero-shot"],
                        linestyle=(0, (5, 3)),
                        linewidth=1.5,
                    )

        for condition_index, (condition, _, color_name) in enumerate(MIX_CONDITIONS):
            payloads = _condition_payloads(run_root, condition)
            for task_index, task in enumerate(TASKS):
                seed_values = [
                    float(value)
                    for payload in payloads
                    if (
                        value := payload["by_group"]["task"]
                        .get(task, {})
                        .get(metric_name)
                    )
                    is not None
                ]
                bootstrap_values = _condition_bootstrap_values(
                    run_root,
                    condition,
                    metric_name,
                    task=task,
                )
                if seed_values:
                    _bootstrap_violin(
                        axis,
                        task_index + offsets[condition_index],
                        bootstrap_values,
                        seed_values,
                        COLORS[color_name],
                    )

        _style_axis(
            axis,
            title,
            limits,
            title_fontsize=source_fontsize,
            title_fontweight="normal",
            title_pad=20,
        )
        axis.set_xticks(
            range(len(TASKS)),
            compact_task_labels,
            fontsize=source_tick_fontsize,
        )
        axis.tick_params(axis="y", labelsize=source_tick_fontsize)
    axes[0].set_ylabel("Robometer", fontsize=source_fontsize)

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
                    color=COLORS[color_name],
                    marker="D",
                    label=label,
                )
                for _, label, color_name in MIX_CONDITIONS
            ],
        ],
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.53, 0.99),
        fontsize=source_fontsize,
    )

    output = run_root / "figures/live/control_mix_mean.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.png")
    figure.savefig(temporary, dpi=300, facecolor="white")
    temporary.replace(output)
    figure.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)


def scaling_figure(run_root: Path, *, last: bool) -> None:
    suffix = "last" if last else "mean"
    metric_specs = (
        (
            f"kendall_tau_{suffix}",
            rf"Kendall $\tau_{{\mathrm{{{suffix}}}}}$" if last else r"$\tau_{\mathrm{mean}}$",
            (-0.2, 1.0),
        ),
        (
            f"extreme_reward_difference_{suffix}",
            f"Extreme {suffix}-progress gap" if last else r"$\Delta$",
            (-0.1, 1.0),
        ),
        (f"roc_auc_{suffix}", "Success ROC-AUC" if last else "ROC-AUC", (0.0, 1.0)),
    )
    zero = _read_metrics(run_root / "eval/zero_shot/paper_metrics.json")
    source_order = (
        ("Teleop", "Rollout", "Human", "Sim HiFi", "Sim LoFi")
        if last
        else ("Teleop", "Rollout", "Human", "Sim HiFi")
    )
    offsets = dict(zip(source_order, np.linspace(-0.24, 0.24, len(source_order)), strict=True))
    paper_style = not last
    source_fontsize = 7.5 / (5.5 / 15.5)
    source_tick_fontsize = 6.5 / (5.5 / 15.5)
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.8 if paper_style else 4.2))
    if paper_style:
        figure.subplots_adjust(
            left=0.07,
            right=0.99,
            bottom=0.20,
            top=0.76,
            wspace=0.18,
        )
    for metric_name, title, limits, axis in (
        (*spec, axis) for spec, axis in zip(metric_specs, axes, strict=True)
    ):
        if zero is not None:
            baseline = zero["metrics"][metric_name]["value"]
            if baseline is not None:
                axis.axhline(
                    baseline,
                    color=COLORS["Zero-shot"],
                    linestyle=(0, (5, 3)),
                    linewidth=1.4,
                )
        for source in source_order:
            x_values: list[float] = []
            centers: list[float] = []
            for condition, (condition_source, demos) in CONDITIONS.items():
                if condition_source != source:
                    continue
                payloads = _condition_payloads(run_root, condition)
                seed_values = [
                    float(payload["metrics"][metric_name]["value"])
                    for payload in payloads
                    if payload["metrics"][metric_name]["value"] is not None
                ]
                if not seed_values:
                    continue
                bootstrap_values = _condition_bootstrap_values(
                    run_root,
                    condition,
                    metric_name,
                )
                x = DEMO_COUNTS.index(demos) + offsets[source]
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
        _style_axis(
            axis,
            title,
            limits,
            title_fontsize=source_fontsize if paper_style else 10,
            title_fontweight="normal" if paper_style else "bold",
            title_pad=20 if paper_style else None,
        )
        axis.set_xticks(
            range(len(DEMO_COUNTS)),
            [str(value) for value in DEMO_COUNTS],
            fontsize=source_tick_fontsize if paper_style else None,
        )
        axis.tick_params(
            axis="y",
            labelsize=source_tick_fontsize if paper_style else None,
        )
        axis.set_xlabel(
            "Training demonstrations per task",
            fontsize=source_fontsize if paper_style else None,
        )
    axes[0].set_ylabel(
        "Robometer" if paper_style else "Metric value",
        fontsize=source_fontsize if paper_style else None,
    )
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
                for source in source_order
            ],
        ],
        loc="upper center",
        ncol=len(source_order) + 1,
        frameon=False,
        bbox_to_anchor=(0.5, 0.99 if paper_style else 1.03),
        fontsize=source_fontsize if paper_style else None,
    )
    output = run_root / "figures/live" / f"scaling_{suffix}.png"
    if paper_style:
        temporary = output.with_name(f".{output.name}.tmp.png")
        figure.savefig(temporary, dpi=300, facecolor="white")
        temporary.replace(output)
        figure.savefig(output.with_suffix(".pdf"), facecolor="white")
        plt.close(figure)
    else:
        figure.tight_layout(rect=(0, 0, 1, 0.91))
        _save(figure, output)


def mae_figure(run_root: Path) -> None:
    metric_specs = (
        ("mae_last", "Partial-credit MAE (last)", (0.0, 0.6)),
        ("mae_mean", "Partial-credit MAE (mean)", (0.0, 0.6)),
        (
            "max_progress_minus_partial_credit",
            "Max progress − partial credit",
            (-0.4, 0.6),
        ),
    )
    source_order = ("Teleop", "Rollout", "Human", "Sim HiFi", "Sim LoFi")
    offsets = dict(zip(source_order, np.linspace(-0.24, 0.24, len(source_order)), strict=True))
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.2))
    zero = _read_metrics(run_root / "eval/zero_shot/paper_metrics.json")
    for metric_name, title, limits, axis in (
        (*spec, axis) for spec, axis in zip(metric_specs, axes, strict=True)
    ):
        if zero is not None:
            baseline = zero["metrics"][metric_name]["value"]
            if baseline is not None:
                axis.axhline(
                    baseline,
                    color=COLORS["Zero-shot"],
                    linestyle=(0, (5, 3)),
                    linewidth=1.4,
                )
        for source in source_order:
            x_values: list[float] = []
            centers: list[float] = []
            for condition, (condition_source, demos) in CONDITIONS.items():
                if condition_source != source:
                    continue
                payloads = _condition_payloads(run_root, condition)
                seed_values = [
                    float(payload["metrics"][metric_name]["value"])
                    for payload in payloads
                    if payload["metrics"][metric_name]["value"] is not None
                ]
                if not seed_values:
                    continue
                bootstrap_values = _condition_bootstrap_values(
                    run_root,
                    condition,
                    metric_name,
                )
                x = DEMO_COUNTS.index(demos) + offsets[source]
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
                )
        _style_axis(axis, title, limits)
        axis.set_xticks(range(len(DEMO_COUNTS)), [str(value) for value in DEMO_COUNTS])
        axis.set_xlabel("Training demonstrations per task")
    axes[0].set_ylabel("Error / signed difference")
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
                for source in source_order
            ],
        ],
        loc="upper center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, 1.03),
    )
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    _save(figure, run_root / "figures/live/mae_and_max_difference.png")


def experiment_status_table(run_root: Path) -> None:
    rows: list[dict[str, str | int]] = []
    robometer_runs = (
        ("zero_shot", (42,)),
        *((condition, SEEDS) for condition in CONDITIONS),
        *((condition, SEEDS) for condition, _, _ in MIX_CONDITIONS),
    )
    for experiment, expected_seeds in robometer_runs:
        completed = [
            seed
            for seed in expected_seeds
            if (
                run_root / "eval"
                / (
                    "zero_shot"
                    if experiment == "zero_shot"
                    else f"{experiment}_seed{seed}"
                )
                / "paper_metrics.json"
            ).is_file()
        ]
        rows.append(
            {
                "model": "Robometer",
                "experiment": experiment,
                "completed_seeds": ",".join(str(seed) for seed in completed),
                "completed_count": len(completed),
                "expected_count": len(expected_seeds),
                "status": (
                    "complete"
                    if len(completed) == len(expected_seeds)
                    else "in progress" if completed else "pending"
                ),
            }
        )

    for experiment in GVL_CONDITIONS:
        completed = [
            seed
            for seed in SEEDS
            if (
                run_root
                / "gvl/runs"
                / f"{experiment}_seed{seed}"
                / "paper_metrics.json"
            ).is_file()
        ]
        rows.append(
            {
                "model": "GVL",
                "experiment": experiment,
                "completed_seeds": ",".join(str(seed) for seed in completed),
                "completed_count": len(completed),
                "expected_count": len(SEEDS),
                "status": (
                    "complete"
                    if len(completed) == len(SEEDS)
                    else "in progress" if completed else "pending"
                ),
            }
        )

    output_dir = run_root / "figures/live"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "experiment_status.csv"
    temporary_csv = csv_path.with_suffix(".csv.tmp")
    fieldnames = (
        "model",
        "experiment",
        "completed_seeds",
        "completed_count",
        "expected_count",
        "status",
    )
    with temporary_csv.open("w", encoding="utf-8", newline="") as table:
        writer = csv.DictWriter(table, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary_csv.replace(csv_path)

    payload = {
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "rows": rows,
    }
    json_path = output_dir / "experiment_status.json"
    temporary_json = json_path.with_suffix(".json.tmp")
    temporary_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary_json.replace(json_path)


def render(run_root: Path) -> None:
    scaling_figure(run_root, last=False)
    combined_control_mean_figure(run_root)
    mixed_source_mean_figure(run_root)
    control_figure(run_root, last=True)
    scaling_figure(run_root, last=True)
    mae_figure(run_root)
    status = {
        condition: [
            seed
            for seed in SEEDS
            if (run_root / "eval" / f"{condition}_seed{seed}" / "paper_metrics.json").is_file()
        ]
        for condition in CONDITIONS
    }
    status.update({
        condition: [
            seed
            for seed in SEEDS
            if (
                run_root
                / "eval"
                / f"{condition}_seed{seed}"
                / "paper_metrics.json"
            ).is_file()
        ]
        for condition, _, _ in MIX_CONDITIONS
    })
    status["zero_shot"] = (
        [42] if (run_root / "eval/zero_shot/paper_metrics.json").is_file() else []
    )
    status_path = run_root / "figures/live/status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    experiment_status_table(run_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--interval-seconds", type=float, default=60.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    while True:
        try:
            render(args.run_root.resolve())
            print(f"[refreshed] {args.run_root / 'figures/live'}", flush=True)
        except Exception as error:
            print(f"[refresh-failed] {type(error).__name__}: {error}", flush=True)
        if args.once:
            return
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
