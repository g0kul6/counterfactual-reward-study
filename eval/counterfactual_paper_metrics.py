"""Compute the offline benchmark metrics defined in the counterfactual paper."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score


def _pairwise_reward_differences(
    records: Sequence[dict[str, Any]],
    terminal_score: np.ndarray,
    mean_score: np.ndarray,
) -> tuple[float | None, float | None, int]:
    """Average high-minus-low prediction margin over same-task, unequal-credit pairs."""
    partial_credit = np.asarray([record["partial_success"] for record in records], dtype=float)
    tasks = np.asarray([str(record["task"]) for record in records])
    scores = np.column_stack((terminal_score, mean_score))
    difference_sum = np.zeros(2, dtype=float)
    pair_count = 0

    for task in np.unique(tasks):
        task_mask = tasks == task
        task_credit = partial_credit[task_mask]
        task_scores = scores[task_mask]
        levels = np.unique(task_credit)
        for high_index in range(1, len(levels)):
            high_scores = task_scores[task_credit == levels[high_index]]
            for low_index in range(high_index):
                low_scores = task_scores[task_credit == levels[low_index]]
                high_count = len(high_scores)
                low_count = len(low_scores)
                difference_sum += (
                    low_count * high_scores.sum(axis=0)
                    - high_count * low_scores.sum(axis=0)
                )
                pair_count += high_count * low_count

    if pair_count == 0:
        return None, None, 0
    differences = difference_sum / pair_count
    return float(differences[0]), float(differences[1]), pair_count


def _extreme_reward_differences(
    records: Sequence[dict[str, Any]],
    terminal_score: np.ndarray,
    mean_score: np.ndarray,
) -> tuple[float | None, float | None, int]:
    """Average per-task gap between highest- and lowest-credit groups."""
    partial_credit = np.asarray([record["partial_success"] for record in records], dtype=float)
    tasks = np.asarray([str(record["task"]) for record in records])
    scores = np.column_stack((terminal_score, mean_score))
    task_gaps: list[np.ndarray] = []

    for task in np.unique(tasks):
        task_mask = tasks == task
        task_credit = partial_credit[task_mask]
        if np.unique(task_credit).size < 2:
            continue
        task_scores = scores[task_mask]
        highest = task_scores[task_credit == np.max(task_credit)].mean(axis=0)
        lowest = task_scores[task_credit == np.min(task_credit)].mean(axis=0)
        task_gaps.append(highest - lowest)

    if not task_gaps:
        return None, None, 0
    differences = np.asarray(task_gaps).mean(axis=0)
    return float(differences[0]), float(differences[1]), len(task_gaps)


def _point_metrics(records: Sequence[dict[str, Any]]) -> dict[str, float | None]:
    partial_credit = np.asarray([record["partial_success"] for record in records], dtype=float)
    terminal_score = np.asarray([record["progress_pred"][-1] for record in records], dtype=float)
    mean_score = np.asarray([np.mean(record["progress_pred"]) for record in records], dtype=float)
    max_score = np.asarray([np.max(record["progress_pred"]) for record in records], dtype=float)
    success = partial_credit == 1.0

    tau_last = kendalltau(partial_credit, terminal_score, variant="b").statistic
    tau_mean = kendalltau(partial_credit, mean_score, variant="b").statistic
    difference_last, difference_mean, _ = _pairwise_reward_differences(records, terminal_score, mean_score)
    extreme_last, extreme_mean, _ = _extreme_reward_differences(records, terminal_score, mean_score)
    auc_last = roc_auc_score(success, terminal_score) if np.unique(success).size == 2 else None
    auc_mean = roc_auc_score(success, mean_score) if np.unique(success).size == 2 else None
    return {
        "kendall_tau_last": None if np.isnan(tau_last) else float(tau_last),
        "kendall_tau_mean": None if np.isnan(tau_mean) else float(tau_mean),
        "mean_reward_difference_last": difference_last,
        "mean_reward_difference_mean": difference_mean,
        "extreme_reward_difference_last": extreme_last,
        "extreme_reward_difference_mean": extreme_mean,
        "mae_last": float(np.mean(np.abs(terminal_score - partial_credit))),
        "mae_mean": float(np.mean(np.abs(mean_score - partial_credit))),
        "mae_max": float(np.mean(np.abs(max_score - partial_credit))),
        "max_progress_minus_partial_credit": float(np.mean(max_score - partial_credit)),
        "roc_auc_last": None if auc_last is None else float(auc_last),
        "roc_auc_mean": None if auc_mean is None else float(auc_mean),
        # Backward-compatible aliases used by older plotting scripts.
        "pc_mae": float(np.mean(np.abs(terminal_score - partial_credit))),
        "roc_auc": None if auc_last is None else float(auc_last),
    }


def _bootstrap(records: Sequence[dict[str, Any]], *, samples: int, seed: int) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {
        "kendall_tau_last": [],
        "kendall_tau_mean": [],
        "mean_reward_difference_last": [],
        "mean_reward_difference_mean": [],
        "extreme_reward_difference_last": [],
        "extreme_reward_difference_mean": [],
        "mae_last": [],
        "mae_mean": [],
        "mae_max": [],
        "max_progress_minus_partial_credit": [],
        "roc_auc_last": [],
        "roc_auc_mean": [],
        "pc_mae": [],
        "roc_auc": [],
    }
    for _ in range(samples):
        indices = rng.integers(0, len(records), size=len(records))
        metrics = _point_metrics([records[index] for index in indices])
        for name, value in metrics.items():
            if value is not None:
                values[name].append(value)
    return {
        name: [float(value) for value in np.quantile(metric_values, [0.025, 0.975])]
        for name, metric_values in values.items()
    }


def summarize(records: Sequence[dict[str, Any]], *, bootstrap_samples: int, seed: int) -> dict[str, Any]:
    if not records:
        raise ValueError("No prediction records were provided")
    if len({record["id"] for record in records}) != len(records):
        raise ValueError("Prediction records must have unique trajectory IDs")

    overall = _point_metrics(records)
    intervals = _bootstrap(records, samples=bootstrap_samples, seed=seed)
    metrics = {name: {"value": value, "ci95": intervals[name]} for name, value in overall.items()}
    terminal_score = np.asarray([record["progress_pred"][-1] for record in records], dtype=float)
    mean_score = np.asarray([np.mean(record["progress_pred"]) for record in records], dtype=float)
    _, _, pair_count = _pairwise_reward_differences(records, terminal_score, mean_score)
    _, _, extreme_task_count = _extreme_reward_differences(records, terminal_score, mean_score)

    groups: dict[str, dict[str, dict[str, float | None]]] = {}
    for field in ("data_source", "task"):
        group_values = sorted({str(record[field]) for record in records})
        groups[field] = {
            value: _point_metrics([record for record in records if str(record[field]) == value])
            for value in group_values
        }

    return {
        "n_episodes": len(records),
        "same_task_ranked_pairs": pair_count,
        "extreme_task_comparisons": extreme_task_count,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "metrics": metrics,
        "by_group": groups,
        "definitions": {
            "kendall_tau_last": "Kendall tau-b between the last progress prediction and human partial credit.",
            "kendall_tau_mean": "Kendall tau-b between mean trajectory progress and human partial credit.",
            "mean_reward_difference_last": (
                "Mean last-progress difference score(a)-score(b) over same-task pairs with credit(a)>credit(b)."
            ),
            "mean_reward_difference_mean": (
                "Mean trajectory-average difference score(a)-score(b) over same-task pairs with credit(a)>credit(b)."
            ),
            "extreme_reward_difference_last": (
                "Mean across tasks of last-progress(highest-credit group)-last-progress(lowest-credit group)."
            ),
            "extreme_reward_difference_mean": (
                "Mean across tasks of trajectory-average(highest-credit group)-trajectory-average(lowest-credit group)."
            ),
            "mae_last": "Mean absolute error between last model score and human partial credit.",
            "mae_mean": "Mean absolute error between mean trajectory score and human partial credit.",
            "mae_max": "Mean absolute error between maximum trajectory score and human partial credit.",
            "max_progress_minus_partial_credit": (
                "Mean signed maximum predicted progress minus human partial credit."
            ),
            "roc_auc_last": (
                "Success/failure ROC-AUC using last model score; success means partial credit 1.0."
            ),
            "roc_auc_mean": (
                "Success/failure ROC-AUC using mean trajectory score; success means partial credit 1.0."
            ),
            "pc_mae": "Backward-compatible alias for mae_last.",
            "roc_auc": "Backward-compatible alias for roc_auc_last.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = json.loads(args.results.read_text(encoding="utf-8"))
    summary = summarize(
        records,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["metrics"], indent=2))


if __name__ == "__main__":
    main()
