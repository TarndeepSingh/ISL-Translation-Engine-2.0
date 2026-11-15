"""Add local SHA-256 and byte size to a canonical manifest copy."""
from __future__ import annotations

import argparse
import os

from .canonical import ManifestError, read_manifest, resolve_media_path, write_manifest
from .validate import sha256_file


def fingerprint_manifest(rows, data_root: str):
    result = []
    for source in rows:
        row = dict(source)
        path = resolve_media_path(data_root, row["media_uri"])
        if not os.path.isfile(path):
            raise ManifestError(f"media file is missing: {row['media_uri']}")
        size = os.path.getsize(path)
        if size <= 0:
            raise ManifestError(f"media file is empty: {row['media_uri']}")
        row["sha256"] = sha256_file(path)
        row["size_bytes"] = str(size)
        row["media_status"] = "present_hashed"
        result.append(row)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", required=True, help="Write a new manifest; source is never modified")
    args = parser.parse_args(argv)
    try:
        rows = read_manifest(args.manifest)
        fingerprinted = fingerprint_manifest(rows, args.data_root)
        write_manifest(fingerprinted, args.out)
    except (OSError, ManifestError) as exc:
        parser.exit(2, f"fingerprinting blocked: {exc}\n")
    print(f"fingerprinted {len(fingerprinted)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
