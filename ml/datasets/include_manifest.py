"""Build a canonical INCLUDE manifest, a download keep-list and an audit report from card metadata.

This is the missing link between the Hugging Face metadata (``ai4bharat/INCLUDE``: label, path,
``include_50``, official split) and the canonical tools (``fingerprint`` -> ``validate`` ->
``preprocess``). It never downloads media and never approves rights: ``--rights-status`` defaults to
``review_pending`` and must be set deliberately after the source terms have been reviewed.

Typical use on Colab (metadata only is 122 kB; videos come later, from Zenodo):

    python -m ml.datasets.include_manifest --write-keep /content/isl-data/keep.txt \\
        --report /content/isl-data/reports/include_audit.json            # no manifest yet

    python -m ml.datasets.include_manifest --data-root /content/isl-data/raw \\
        --rights-status approved_research --split-policy official \\
        --out /content/isl-data/manifests/include50.csv \\
        --report /content/isl-data/reports/include_manifest.json

Split policies
  official       the benchmark's own video-level split. It shares recording sessions across splits, so
                 scores can over-estimate performance on new signers; use for comparison.
  session-proxy  whole proxy sessions (inferred from camera file numbering, see
                 ``ml.datasets.manifest.infer_sessions``) go to one split. This is a heuristic proxy
                 for signer/session, not signer identity; inspect the report before trusting it.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from . import manifest as legacy
from .adapters import adapt_include_records, audit_include_records, include_path_label_consistent
from .canonical import RIGHTS_STATES, validate_row_shape, write_manifest

DEFAULT_VERSION = "zenodo-4010759-v1"
SPLIT_NAMES = {"validation": "val", "valid": "val", "dev": "val"}


REQUIRED_COLUMNS = {"label", "video_path"}


def _checked(records: list[dict[str, Any]], origin: str) -> list[dict[str, Any]]:
    if not records:
        raise SystemExit(f"{origin}: no rows found")
    missing = REQUIRED_COLUMNS - set(records[0])
    if missing:
        raise SystemExit(f"{origin}: expected columns {sorted(REQUIRED_COLUMNS)} but the first row has "
                         f"{sorted(records[0])}; missing {sorted(missing)}. Send this message to the maintainer.")
    return records


def load_records(metadata: str | None) -> list[dict[str, Any]]:
    """Load card rows with a ``split`` field from a local JSON/CSV file or the HF Hub."""
    if metadata is None:
        try:
            from datasets import load_dataset  # lazy: needs network on first use
            dataset = load_dataset("ai4bharat/INCLUDE")
        except Exception as exc:  # noqa: BLE001 - turn any hub/network/format failure into guidance
            raise SystemExit(
                f"could not load ai4bharat/INCLUDE from the Hugging Face Hub: {type(exc).__name__}: {exc}\n"
                "Fallback: open https://huggingface.co/datasets/ai4bharat/INCLUDE in a browser, download the "
                "metadata file(s) from the 'Files' tab, put them in Drive, convert/merge to one .json or .csv "
                "with a 'split' column, and rerun with --metadata FILE.") from exc
        records = [dict(row, split=SPLIT_NAMES.get(name, name)) for name, part in dataset.items() for row in part]
        return _checked(records, "ai4bharat/INCLUDE (Hub)")
    path = Path(metadata)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            records = [dict(row, split=SPLIT_NAMES.get(name, name)) for name, rows in data.items() for row in rows]
        else:
            records = [dict(row) for row in data]
        return _checked(records, str(path))
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as stream:
            return _checked([dict(row) for row in csv.DictReader(stream)], str(path))
    raise SystemExit(f"unsupported metadata file type: {path.suffix} (use .json or .csv)")


def _is_include_50(record: dict[str, Any]) -> bool:
    value = record.get("include_50", False)
    return value if isinstance(value, bool) else str(value).strip().lower() in {"1", "true", "yes", "y"}


def assign_session_proxy_splits(rows: list[dict[str, str]], seed: int) -> None:
    """Overwrite split with whole-proxy-session assignment; sets session_id/group_id/split_policy."""
    view = [{"dataset": "include", "path": row["media_uri"], "session_id": "", "sample_id": row["sample_id"],
             "split": ""} for row in rows]
    legacy.infer_sessions(view)
    for row, item in zip(rows, view):
        row["session_id"] = item["session_id"]
        row["group_id"] = f"session:{item['session_id']}"
    legacy.make_group_split(view, seed=seed)
    for row, item in zip(rows, view):
        row["split"] = item["split"]
        row["split_policy"] = "session_proxy_grouped"


def assign_balanced_official_splits(rows: list[dict[str, str]], seed: int,
                                    fill: tuple[tuple[str, int], ...] = (("test", 3), ("val", 2)),
                                    keep_in_train: int = 5) -> dict[str, dict[str, list[str]]]:
    """Keep the official video-level split exactly, except for words that are MISSING from val or test.

    A word with zero videos in a split gets ``n`` videos moved in from train (seeded, deterministic, whole
    groups only, never leaving fewer than ``keep_in_train`` in train). Words already present are untouched.
    Returns {split: {label: [moved sample_ids]}} for the report. Like the official split this is video-level
    and can share recording sessions across splits - it is NOT signer-independent.
    """
    import random
    rng = random.Random(seed)
    moved: dict[str, dict[str, list[str]]] = {}
    labels = sorted({row["label_text"] for row in rows})
    for target, amount in fill:
        for label in labels:
            of_label = [row for row in rows if row["label_text"] == label]
            if any(row["split"] == target for row in of_label):
                continue
            by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in of_label:
                if row["split"] == "train":
                    by_group[row["group_id"] or row["sample_id"]].append(row)
            keys = sorted(by_group)
            rng.shuffle(keys)
            need = amount
            for key in keys:
                group = by_group[key]
                train_left = sum(r["split"] == "train" for r in of_label) - len(group)
                if need <= 0 or train_left < keep_in_train:
                    continue
                for row in group:
                    row["split"] = target
                    row["split_policy"] = "official_balanced"
                    moved.setdefault(target, {}).setdefault(label, []).append(row["sample_id"])
                need -= len(group)
    return moved


def coverage_report(rows: list[dict[str, str]]) -> dict[str, Any]:
    per_split: Counter[str] = Counter(row["split"] for row in rows)
    labels_by_split: dict[str, set[str]] = defaultdict(set)
    groups_by_split: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        labels_by_split[row["split"]].add(row["label_text"])
        if row["group_id"]:
            groups_by_split[row["group_id"]].add(row["split"])
    train = labels_by_split.get("train", set())
    return {
        "rows_per_split": dict(per_split),
        "classes": len({row["label_text"] for row in rows}),
        "classes_missing_from_train": sorted(
            {label for split in ("val", "test") for label in labels_by_split.get(split, set())} - train),
        "groups_in_multiple_splits": sum(1 for splits in groups_by_split.values() if len(splits) > 1),
        "rows_per_class_min_max": [
            min(Counter(r["label_text"] for r in rows).values(), default=0),
            max(Counter(r["label_text"] for r in rows).values(), default=0)],
    }


def build(records: list[dict[str, Any]], *, include_50_only: bool, split_policy: str, data_root: str | None,
          dataset_version: str, license_id: str, rights_status: str, seed: int) -> tuple[list[dict[str, str]], dict]:
    report: dict[str, Any] = {"audit_all_rows": audit_include_records(records)}
    selected = [r for r in records if _is_include_50(r)] if include_50_only else list(records)
    report["selection"] = {"include_50_only": include_50_only, "rows_selected": len(selected),
                           "audit_selected_rows": audit_include_records(selected)}
    rows = adapt_include_records(
        selected, dataset_version=dataset_version, license_id=license_id, rights_status=rights_status,
        use_official_splits=(split_policy in {"official", "balanced"}), exclude_label_path_mismatch=True)
    report["rows_after_label_path_exclusion"] = len(rows)
    seen: dict[str, dict[str, str]] = {}
    duplicates = 0
    conflicts: list[str] = []
    for row in rows:
        first = seen.get(row["media_uri"])
        if first is None:
            seen[row["media_uri"]] = row
        elif first["label_text"] != row["label_text"]:
            conflicts.append(f"{row['media_uri']}: {first['label_text']!r} vs {row['label_text']!r}")
        else:
            duplicates += 1
    if conflicts:   # never let the first row silently win
        raise SystemExit(f"{len(conflicts)} videos are listed under conflicting labels the path check cannot "
                         f"resolve; fix the metadata. First: {conflicts[:3]}")
    rows = list(seen.values())
    report["duplicate_media_rows_dropped"] = duplicates
    if data_root:
        root = Path(data_root)
        present = [row for row in rows if (root / row["media_uri"]).is_file()]
        report["media"] = {"data_root": str(root), "listed": len(rows), "present": len(present),
                           "missing_on_disk": len(rows) - len(present)}
        rows = present
        for row in rows:
            row["media_status"] = "present"
    if split_policy == "session-proxy":
        assign_session_proxy_splits(rows, seed)
    if split_policy == "balanced":
        report["balanced_moves"] = assign_balanced_official_splits(rows, seed)
    report["split_policy"] = split_policy
    report["coverage"] = coverage_report(rows)
    schema_issues = [issue for row in rows for issue in validate_row_shape(row)]
    report["schema_issues"] = len(schema_issues)
    report["rights"] = {"license_id": license_id, "rights_status": rights_status,
                        "note": "recorded from the CLI argument; this tool does not review terms"}
    return rows, report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", help="local .json/.csv card metadata; default loads ai4bharat/INCLUDE from the Hub")
    ap.add_argument("--all-classes", action="store_true", help="use all 263 classes instead of INCLUDE-50")
    ap.add_argument("--data-root", help="keep only rows whose media exists under this root")
    ap.add_argument("--split-policy", choices=("official", "session-proxy", "balanced"), default="official")
    ap.add_argument("--dataset-version", default=DEFAULT_VERSION)
    ap.add_argument("--license-id", default="CC-BY-4.0")
    ap.add_argument("--rights-status", default="review_pending", choices=sorted(RIGHTS_STATES))
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", help="canonical manifest CSV to write")
    ap.add_argument("--write-keep", help="write the selected video paths (for isl.download_include --keep)")
    ap.add_argument("--report", help="write the JSON audit/coverage report here")
    args = ap.parse_args(argv)
    records = load_records(args.metadata)
    rows, report = build(
        records, include_50_only=not args.all_classes, split_policy=args.split_policy, data_root=args.data_root,
        dataset_version=args.dataset_version, license_id=args.license_id, rights_status=args.rights_status,
        seed=args.seed)
    if args.write_keep:
        Path(args.write_keep).parent.mkdir(parents=True, exist_ok=True)
        Path(args.write_keep).write_text("\n".join(sorted(row["media_uri"] for row in rows)) + "\n", encoding="utf-8")
        report["keep_file"] = {"path": args.write_keep, "entries": len(rows)}
    if args.out:
        if report["schema_issues"]:
            raise SystemExit(f"refusing to write manifest: {report['schema_issues']} schema issues (see --report)")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        write_manifest(rows, args.out)
        report["manifest"] = {"path": args.out, "rows": len(rows)}
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


__all__ = ["build", "load_records", "main", "include_path_label_consistent"]

if __name__ == "__main__":
    sys.exit(main())
