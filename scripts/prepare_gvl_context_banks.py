#!/usr/bin/env python3
"""Build deterministic, task-specific GVL terminal-reward context banks."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SOURCE_NAMES = {
    "teleop": "kinesthetic_data",
    "rollout": "rollouts_pi0",
    "human": "human_demo_data",
    "hifi": "sim_hifi",
}
DEFAULT_ANNOTATIONS = "annotations/counterfactual_annotations.jsonl"
SELECTION_VERSION = "reward-duration-diversity-v1"


def normalize_task(task: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", task.lower())).strip("_")


def duration_seconds(record: dict[str, Any]) -> float:
    start = int(record.get("raw_start_frame") or 0)
    end_value = record.get("end_frame")
    end = int(end_value) if end_value is not None else int(record["frame_count"]) - 1
    return max(0.0, (end - start + 1) / float(record["fps"]))


def _unique_valid(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.get("ambiguous") or record.get("invalid_media"):
            continue
        trajectory_id = str(record["trajectory_id"])
        current = by_id.get(trajectory_id)
        if current is None or int(record.get("revision", 0)) > int(current.get("revision", 0)):
            by_id[trajectory_id] = record
    return sorted(by_id.values(), key=lambda item: item["trajectory_id"])


def _diverse_failures(failures: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    by_reward: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for failure in failures:
        by_reward[float(failure["reward"])].append(failure)
    for group in by_reward.values():
        group.sort(key=lambda item: (duration_seconds(item), item["trajectory_id"]))

    selected: list[dict[str, Any]] = []
    reward_levels = sorted(by_reward)
    if count >= len(reward_levels):
        covered_levels = reward_levels
    elif count == 1:
        covered_levels = [reward_levels[len(reward_levels) // 2]]
    else:
        level_indices = [
            round(index * (len(reward_levels) - 1) / (count - 1))
            for index in range(count)
        ]
        covered_levels = [reward_levels[index] for index in level_indices]

    for index, reward in enumerate(covered_levels):
        group = by_reward[reward]
        candidate = group[0] if index % 2 == 0 else group[-1]
        candidate = dict(candidate)
        candidate["selection_reason"] = f"reward_coverage_{reward:g}"
        selected.append(candidate)

    remaining = [
        record
        for record in failures
        if record["trajectory_id"] not in {item["trajectory_id"] for item in selected}
    ]
    max_duration = max((duration_seconds(item) for item in failures), default=1.0) or 1.0
    while remaining and len(selected) < count:
        def diversity_score(candidate: dict[str, Any]) -> tuple[float, str]:
            candidate_point = (float(candidate["reward"]), duration_seconds(candidate) / max_duration)
            distance = min(
                abs(candidate_point[0] - float(chosen["reward"]))
                + abs(candidate_point[1] - duration_seconds(chosen) / max_duration)
                for chosen in selected
            )
            return distance, candidate["trajectory_id"]

        candidate = max(remaining, key=diversity_score)
        candidate = dict(candidate)
        candidate["selection_reason"] = "reward_duration_farthest"
        selected.append(candidate)
        remaining = [
            item for item in remaining if item["trajectory_id"] != candidate["trajectory_id"]
        ]
    return selected


def select_demonstrations(
    records: Iterable[dict[str, Any]],
    *,
    limit: int = 5,
    use_all_if_under_limit: bool = False,
) -> list[dict[str, Any]]:
    """Select at most one success and reward/duration-diverse failures."""
    unique = _unique_valid(records)
    successes = [record for record in unique if float(record["reward"]) == 1.0]
    failures = [record for record in unique if float(record["reward"]) < 1.0]
    success = min(successes, key=lambda item: item["trajectory_id"]) if successes else None

    if use_all_if_under_limit and len(unique) < limit and len(successes) <= 1:
        selected = ([success] if success else []) + failures
        return [
            {**record, "selection_reason": "all_unique_under_limit"}
            for record in selected
        ]

    selected: list[dict[str, Any]] = []
    if success is not None:
        selected.append({**success, "selection_reason": "single_success"})
    selected.extend(_diverse_failures(failures, limit - len(selected)))
    return selected


def _left_camera_path(video_path: str) -> str:
    path = Path(video_path)
    if path.name in {"mosaic.mp4", "combined.mp4"}:
        candidate = path.with_name("left.mp4")
    elif path.name.endswith("_combined.mp4"):
        candidate = path.with_name(
            path.name.removesuffix("_combined.mp4") + "_left.mp4"
        )
    else:
        candidate = path
    if candidate != path and not candidate.is_file():
        raise FileNotFoundError(f"Left-camera context video not found: {candidate}")
    return str(candidate)


def _bank_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "trajectory_id": record["trajectory_id"],
        "video_path": _left_camera_path(record["video_path"]),
        "reward": float(record["reward"]),
        "raw_start_frame": int(record.get("raw_start_frame") or 0),
        "end_frame": record.get("end_frame"),
        "frame_count": int(record["frame_count"]),
        "fps": float(record["fps"]),
        "duration_seconds": duration_seconds(record),
        "selection_reason": record["selection_reason"],
        "rubric_version": record.get("rubric_version"),
    }


def build_bank(
    records: Iterable[dict[str, Any]],
    *,
    source: str,
    num_shots: int,
    preserve_input_order: bool = False,
) -> dict[str, Any]:
    if source not in SOURCE_NAMES:
        raise ValueError(f"source must be one of: {', '.join(sorted(SOURCE_NAMES))}")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        task = str(record.get("task_display_name") or record["task_slug"])
        groups[normalize_task(task)].append(record)

    tasks: dict[str, Any] = {}
    for task_key, task_records in sorted(groups.items()):
        if preserve_input_order:
            seen: set[str] = set()
            selected = []
            for record in task_records:
                trajectory_id = str(record["trajectory_id"])
                if trajectory_id in seen:
                    continue
                seen.add(trajectory_id)
                if record.get("ambiguous") or record.get("invalid_media"):
                    continue
                selected.append(
                    {**record, "selection_reason": "nested_subset_order"}
                )
                if len(selected) == num_shots:
                    break
            if len(selected) != num_shots:
                raise ValueError(
                    f"{task_key}: expected {num_shots} ordered demonstrations, "
                    f"found {len(selected)}"
                )
        else:
            selected = select_demonstrations(
                task_records,
                limit=num_shots,
                use_all_if_under_limit=source == "teleop",
            )
        task_name = str(task_records[0].get("task_display_name") or task_records[0]["task_slug"])
        tasks[task_key] = {
            "task": task_name,
            "demonstrations": [_bank_record(record) for record in selected],
        }

    return {
        "schema_version": 1,
        "selection_version": SELECTION_VERSION,
        "context_source": source,
        "max_demonstrations_per_task": num_shots,
        "label_mode": "trajectory_average_reward",
        "tasks": tasks,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, choices=sorted(SOURCE_NAMES))
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-shots", type=int, default=5)
    parser.add_argument(
        "--preserve-input-order",
        action="store_true",
        help=(
            "Use the annotation file's per-task order directly. This is intended "
            "for nested experiment-subset JSONL files so smaller shot counts are "
            "exact prefixes of larger conditions."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.num_shots < 1:
        parser.error("--num-shots must be positive")

    study_root = Path(__file__).resolve().parents[3]
    annotations = args.annotations or study_root / DEFAULT_ANNOTATIONS
    records = [
        record
        for record in _read_jsonl(annotations)
        if record.get("source") == SOURCE_NAMES[args.source]
        and record.get("split_role") == "train"
    ]
    bank = build_bank(
        records,
        source=args.source,
        num_shots=args.num_shots,
        preserve_input_order=args.preserve_input_order,
    )
    fingerprint = hashlib.sha256(annotations.read_bytes()).hexdigest()
    bank["annotation_path"] = str(annotations.resolve())
    bank["annotation_sha256"] = fingerprint

    summary = {
        task: {
            "count": len(entry["demonstrations"]),
            "rewards": [demo["reward"] for demo in entry["demonstrations"]],
            "durations": [round(demo["duration_seconds"], 3) for demo in entry["demonstrations"]],
        }
        for task, entry in bank["tasks"].items()
    }
    print(json.dumps(summary, indent=2))
    if args.dry_run:
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(bank, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
