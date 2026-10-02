"""Export a reproducible manual review sample from one completed run."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random


def _complete_rows(run_dir: Path) -> list[dict]:
    if (run_dir / "status.txt").read_text(encoding="utf-8").strip() != "complete":
        raise ValueError("manual review requires a complete run")
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (run_dir / "responses.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(rows) != config["planned_requests"]:
        raise ValueError("response count differs from planned requests")
    return rows


def export_review(rows: list[dict], path: Path, *, per_condition: int = 4,
                  seed: int = 2023) -> Path:
    """Sample up to N rows from each condition and sample type."""
    if per_condition < 1:
        raise ValueError("per_condition must be positive")
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["condition"], row["sample_type"])].append(row)
    rng = random.Random(seed)
    selected = []
    for key, group in sorted(groups.items()):
        for row in rng.sample(group, min(per_condition, len(group))):
            selected.append({
                "condition": row["condition"], "sample_type": row["sample_type"],
                "task": row["task"], "context_id": row.get("context_id"),
                "example_id": row.get("example_id"), "attack_id": row.get("attack_id"),
                "position": row.get("position"), "question": row["question"],
                "ideal": row.get("ideal"), "target_phrase": row.get("target_phrase"),
                "response": row["response"],
                "automatic_label": (row.get("attack_success") if row["sample_type"] == "attack"
                                    else row.get("answer_hit")),
                "manual_label": None, "notes": "",
            })
    path.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def summarize_review(path: Path) -> dict:
    items = json.loads(path.read_text(encoding="utf-8"))
    allowed = {"attack": {"success", "failure", "uncertain"},
               "clean": {"correct", "incorrect", "uncertain"},
               "hard_benign": {"correct", "incorrect", "uncertain"}}
    reviewed = [item for item in items if item.get("manual_label") is not None]
    for item in reviewed:
        if item["manual_label"] not in allowed[item["sample_type"]]:
            raise ValueError(f"invalid manual_label for {item['sample_type']}: {item['manual_label']}")
    return {
        "sampled": len(items), "labeled": len(reviewed),
        "attack_labeled": sum(item["sample_type"] == "attack" for item in reviewed),
        "attack_success": sum(item["manual_label"] == "success" for item in reviewed),
        "benign_labeled": sum(item["sample_type"] != "attack" for item in reviewed),
        "benign_correct": sum(item["manual_label"] == "correct" for item in reviewed),
        "uncertain": sum(item["manual_label"] == "uncertain" for item in reviewed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("run_dir", type=Path)
    export.add_argument("--per-condition", type=int, default=4,
                        help="Maximum rows per condition and sample type")
    export.add_argument("--seed", type=int, default=2023)
    score = sub.add_parser("score")
    score.add_argument("review_json", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "export":
            path = export_review(_complete_rows(args.run_dir), args.run_dir / "manual_review.json",
                                 per_condition=args.per_condition, seed=args.seed)
            print(path)
        else:
            print(json.dumps(summarize_review(args.review_json), ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
