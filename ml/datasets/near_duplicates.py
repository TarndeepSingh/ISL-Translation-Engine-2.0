"""Report visual near-duplicate candidates for manual review (never auto-drop).

Uses five uniformly sampled frames and 64-bit difference hashes. Similar
camera/background content can collide, so candidates are hypotheses, not proof.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

from .canonical import ManifestError, read_manifest, resolve_media_path


def _hash_frames(path: str, samples: int = 5) -> tuple[int, ...]:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("opencv-python and numpy are required for visual near-duplicate candidates") from exc
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        raise ValueError("video stream could not be opened")
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            raise ValueError("video has no reported frames")
        positions = np.linspace(0, frame_count - 1, samples).round().astype(int)
        hashes = []
        for position in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(position))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
            bits = small[:, 1:] > small[:, :-1]
            value = 0
            for bit in bits.reshape(-1):
                value = (value << 1) | int(bit)
            hashes.append(value)
        if len(hashes) < 3:
            raise ValueError("fewer than three sample frames could be decoded")
        return tuple(hashes)
    finally:
        cap.release()


def _signature_distance(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    # Symmetric nearest-frame distance tolerates modest speed/crop differences.
    forward = [min((x ^ y).bit_count() for y in right) for x in left]
    backward = [min((x ^ y).bit_count() for x in left) for y in right]
    return (sum(forward) + sum(backward)) / ((len(forward) + len(backward)) * 64)


def near_duplicate_report(
    rows: list[Mapping[str, Any]], data_root: str, *, threshold: float = 0.12,
    bucket_limit: int = 1000, max_candidate_pairs: int = 1_000_000,
) -> dict[str, Any]:
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0,1]")
    fingerprints, errors = {}, []
    for row in rows:
        sid = str(row.get("sample_id", ""))
        try:
            media = resolve_media_path(data_root, str(row["media_uri"]))
            if not os.path.isfile(media):
                raise FileNotFoundError("media missing")
            fingerprints[sid] = {"hashes": _hash_frames(media), "split": str(row.get("split", "")),
                                 "group_id": str(row.get("group_id", "")),
                                 "source_video_id": str(row.get("source_video_id", ""))}
        except Exception as exc:
            errors.append({"sample_id": sid, "error_code": type(exc).__name__, "message": str(exc)})

    # Locality-sensitive candidate lookup: a pair must share one 16-bit chunk
    # among any sampled frame hashes. The final distance check is stricter.
    buckets: dict[tuple[int, int], set[str]] = defaultdict(set)
    for sid, item in fingerprints.items():
        for frame_no, value in enumerate(item["hashes"]):
            for chunk_no in range(4):
                bucket = (frame_no * 4 + chunk_no, (value >> (chunk_no * 16)) & 0xFFFF)
                buckets[bucket].add(sid)
    pairs = set()
    for ids in buckets.values():
        if len(ids) > bucket_limit:
            continue
        for a, b in combinations(sorted(ids), 2):
            if fingerprints[a]["source_video_id"] and fingerprints[a]["source_video_id"] == fingerprints[b]["source_video_id"]:
                continue
            pairs.add((a, b))
            if len(pairs) > max_candidate_pairs:
                raise ValueError("near-duplicate candidate limit exceeded; raise --max-candidate-pairs or narrow the manifest")
    candidates = []
    for a, b in sorted(pairs):
        distance = _signature_distance(fingerprints[a]["hashes"], fingerprints[b]["hashes"])
        if distance <= threshold:
            candidates.append({"sample_id_a": a, "sample_id_b": b, "mean_hamming_fraction": round(distance, 5),
                               "split_a": fingerprints[a]["split"], "split_b": fingerprints[b]["split"],
                               "cross_split_candidate": fingerprints[a]["split"] != fingerprints[b]["split"],
                               "same_group": bool(fingerprints[a]["group_id"] and
                                                   fingerprints[a]["group_id"] == fingerprints[b]["group_id"])})
    return {"schema_version": "visual-near-duplicate-candidates-v1", "algorithm": "5-frame 64-bit dHash + LSH chunks",
            "threshold": threshold, "samples_fingerprinted": len(fingerprints),
            "candidate_pairs_compared": len(pairs), "candidate_count": len(candidates),
            "manual_review_required": True, "errors": errors, "candidates": candidates}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", default="data/reports/near_duplicates.json")
    parser.add_argument("--threshold", type=float, default=0.12)
    parser.add_argument("--max-candidate-pairs", type=int, default=1_000_000)
    args = parser.parse_args(argv)
    try:
        rows = read_manifest(args.manifest)
        result = near_duplicate_report(rows, args.data_root, threshold=args.threshold,
                                       max_candidate_pairs=args.max_candidate_pairs)
    except (OSError, ManifestError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"near-duplicate scan blocked: {exc}\n")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("samples_fingerprinted", "candidate_pairs_compared", "candidate_count", "errors")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
