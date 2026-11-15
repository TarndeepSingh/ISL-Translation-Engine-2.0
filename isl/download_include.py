import argparse, hashlib, json, os, shutil, stat, time, urllib.request, zipfile
from pathlib import Path, PurePosixPath

from ml.datasets.canonical import ManifestError, safe_relative_uri
from isl.remote_zip import RangeFile, RangeNotSupported

API = os.getenv("ISL_ZENODO_API", "https://zenodo.org/api/records/4010759")


def _checksum_ok(path, checksum):
    if not checksum:
        return True
    try:
        algorithm, expected = checksum.split(":", 1)
        digest = hashlib.new(algorithm.lower())
    except (ValueError, TypeError):
        return False
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().lower() == expected.lower()


def fetch(url, dst, size, checksum="", tries=12):
    """Download with automatic retry + resume (HTTP Range). Returns True on success."""
    part = dst + ".part"
    for t in range(tries):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if size and have == size:
            os.replace(part, dst)
            if _checksum_ok(dst, checksum): return True
            os.remove(dst); return False
        if size and have > size:
            os.remove(part); have = 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                mode = "ab" if (have and r.status == 206) else "wb"
                with open(part, mode) as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
        except Exception as e:
            print(f"  network problem ({type(e).__name__}); retry {t + 1}/{tries}", flush=True)
            time.sleep(min(60, 5 * (t + 1))); continue
        if not size or os.path.getsize(part) == size:
            os.replace(part, dst)
            if _checksum_ok(dst, checksum): return True
            print("  source checksum mismatch; removing incomplete archive", flush=True)
            os.remove(dst)
            if os.path.exists(part): os.remove(part)
            return False
        print(f"  incomplete ({os.path.getsize(part)}/{size} bytes); resuming", flush=True)
    return False


def required_free_bytes(selected, delete_zips=False, margin=1.1):
    """Free space needed before starting. With delete_zips only one archive is on disk at a time
    (extraction headroom is checked per archive in safe_extract_zip); otherwise all are kept."""
    sizes = [size for _, _, size, _ in selected]
    return int((max(sizes, default=0) if delete_zips else sum(sizes)) * margin)


def describe_zip(zip_path, limit=6):
    """(number of files, first few member names): shown when nothing in an archive matched the keep list."""
    with zipfile.ZipFile(zip_path) as archive:
        names = [i.filename for i in archive.infolist() if not i.is_dir()]
    return len(names), names[:limit]


def safe_extract_zip(zip_path, root, keep=None, max_extract_bytes=80 * 1024**3):
    """Extract selected ordinary files without traversal, symlinks or overwrite."""
    root = Path(root).resolve()
    keep_paths = None
    if keep is not None:
        # Members are matched on their last three path parts, case-insensitively, and written at the
        # keep list's own spelling so the result is exactly the manifest's media_uri whatever extra
        # top-level folder or capitalisation the archive uses.
        keep_paths, tails = {}, {}
        for item in keep:
            item = str(item).strip().replace("\\", "/")
            if item:
                keep_paths[item.lower()] = item
                tails.setdefault("/".join(item.lower().split("/")[-2:]), []).append(item)
        # Fallback for archives that lack the category folder: match "label/file" only if it is unambiguous.
        tail_paths = {tail: items[0] for tail, items in tails.items() if len(items) == 1}
    selected = []
    total_bytes = 0
    with zipfile.ZipFile(zip_path) as archive:
        entries = archive.infolist()
        if len(entries) > 500_000:
            raise ValueError("archive contains too many entries")
        for info in entries:
            if info.is_dir():
                continue
            parts = info.filename.replace("\\", "/").split("/")
            canonical = None
            if keep_paths is not None:
                canonical = keep_paths.get("/".join(parts[-3:]).lower())
                if canonical is None and len(parts) >= 2:
                    canonical = tail_paths.get("/".join(parts[-2:]).lower())
                if canonical is None:
                    continue
            rel = safe_relative_uri(canonical if canonical is not None else info.filename)
            mode = (info.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(mode):
                raise ValueError(f"archive entry is a symbolic link: {info.filename}")
            if info.flag_bits & 1:
                raise ValueError(f"encrypted archive entry is not supported: {info.filename}")
            total_bytes += info.file_size
            if total_bytes > max_extract_bytes:
                raise ValueError("selected archive members exceed --max-extract-bytes")
            target = (root / Path(*PurePosixPath(rel).parts)).resolve()
            if os.path.commonpath((str(root), str(target))) != str(root):
                raise ValueError(f"archive entry escapes output root: {info.filename}")
            selected.append((info, target))
        free = shutil.disk_usage(root).free
        if total_bytes > free:
            raise OSError(f"not enough free space to extract selected files ({total_bytes} bytes required)")
        count = 0
        prefetch = getattr(zip_path, "prefetch", None)    # remote archives: one request per member
        for info, target in selected:
            if prefetch is not None:
                prefetch(info.header_offset, info.compress_size + 30 + 2 * len(info.filename) + 1024)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if target.stat().st_size == info.file_size:
                    left, right = hashlib.sha256(), hashlib.sha256()
                    with archive.open(info) as source, open(target, "rb") as existing:
                        while True:
                            a, b = source.read(1 << 20), existing.read(1 << 20)
                            if not a and not b: break
                            left.update(a); right.update(b)
                    if left.digest() == right.digest(): continue
                raise FileExistsError(f"refusing to overwrite existing file: {target}")
            copied = 0
            try:
                with archive.open(info) as source, open(target, "xb") as output:
                    while True:
                        chunk = source.read(1 << 20)
                        if not chunk: break
                        copied += len(chunk)
                        if copied > info.file_size or copied > max_extract_bytes:
                            raise ValueError(f"archive member exceeded declared extraction size: {info.filename}")
                        output.write(chunk)
                if copied != info.file_size:
                    raise ValueError(f"archive member size mismatch: {info.filename}")
            except Exception:
                target.unlink(missing_ok=True)
                raise
            count += 1
    return count


def inspect_archive(zip_file, keep):
    """Describe an archive's layout and how many members the keep list would select, three ways.

    ``exact`` is the rule used for extraction (last three path parts, case-insensitive). ``last2`` and
    ``name_only`` are looser diagnostics: if they find far more than ``exact``, the archive's folder layout
    differs from the metadata's and extraction would silently skip those videos.
    """
    wanted = {str(k).strip().replace("\\", "/").lower() for k in keep if str(k).strip()}
    wanted2 = {"/".join(k.split("/")[-2:]) for k in wanted}
    wanted1 = {k.split("/")[-1] for k in wanted}
    with zipfile.ZipFile(zip_file) as archive:
        names = [i.filename for i in archive.infolist() if not i.is_dir()]
    parts = [n.replace("\\", "/").lower().split("/") for n in names]
    return {
        "members": len(names), "first_names": names[:6],
        "exact": sum("/".join(p[-3:]) in wanted for p in parts),
        "last2": sum("/".join(p[-2:]) in wanted2 for p in parts),
        "name_only": sum(p[-1] in wanted1 for p in parts),
    }


def list_archives(api=None):
    """Return (name, url, size_bytes, checksum) for every file in the Zenodo record."""
    with urllib.request.urlopen(api or API) as r:
        return [(f["key"], f["links"]["self"], int(f.get("size", 0)), f.get("checksum", ""))
                for f in json.load(r)["files"]]


def keep_digest(keep):
    """Short fingerprint of the keep list, so changing it forces a re-extract of finished archives."""
    if keep is None:
        return "all"
    items = sorted(str(k).strip().replace("\\", "/").lower() for k in keep if str(k).strip())
    return hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest()[:16]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true"); ap.add_argument("--all", action="store_true")
    ap.add_argument("--only", nargs="*", default=[]); ap.add_argument("--out", default="data/INCLUDE")
    ap.add_argument("--delete-zips", action="store_true"); ap.add_argument("--keep")
    ap.add_argument("--confirm-reviewed-rights", action="store_true",
                    help="Confirm that this release's media/license terms were reviewed before downloading")
    ap.add_argument("--max-extract-bytes", type=int, default=80 * 1024**3)
    ap.add_argument("--remote", action="store_true",
                    help="extract only the wanted members with HTTP Range requests (no full archive download)")
    ap.add_argument("--inspect", nargs="*", metavar="ARCHIVE",
                    help="list these archives' layout and keep-list matches without downloading them")
    ap.add_argument("--stop-after", type=int, default=0,
                    help="smoke test: process archives smallest-first and stop once this many videos were extracted")
    ap.add_argument("--max-archives", type=int, default=0,
                    help="with --stop-after: give up (exit 2) after trying this many archives without success")
    a = ap.parse_args(argv)
    keep = set(Path(a.keep).read_text(encoding="utf-8").splitlines()) if a.keep else None
    if keep is not None and not any(line.strip() for line in keep):
        raise SystemExit("--keep file is empty; refusing to extract nothing")
    files = list_archives()
    for k, _, size, _ in files:
        print(f"{size / 1e9:6.2f} GB  {k}")
    print(f"total {sum(size for _, _, size, _ in files) / 1e9:.1f} GB in {len(files)} files")
    if a.inspect is not None:
        if keep is None:
            raise SystemExit("--inspect needs --keep so matches can be counted")
        names = a.inspect or [files[0][0]]
        everything = any(n.lower() == "all" for n in names)
        wanted = len({str(k).strip().lower() for k in keep if str(k).strip()})
        totals = {"members": 0, "exact": 0, "last2": 0, "name_only": 0}
        inspected = 0
        for k, url, size, _ in files:
            if not k.lower().endswith(".zip"):
                continue
            if everything or k in names or any(n.lower() in k.lower() for n in names):
                try:
                    rf = RangeFile(url, size, block=1 << 20)
                    info = inspect_archive(rf, keep)
                except (RangeNotSupported, OSError, zipfile.BadZipFile) as exc:
                    print(f"INSPECT UNAVAILABLE for {k}: {type(exc).__name__}: {exc}")
                    return 3
                inspected += 1
                for field in totals:
                    totals[field] += info[field]
                print(f"\n{k}: {info['members']} members; matches by last-3 parts={info['exact']}, "
                      f"last-2={info['last2']}, file name only={info['name_only']} "
                      f"(read {rf.bytes_fetched / 1e6:.2f} MB in {rf.requests} requests)")
                for n in info["first_names"]:
                    print("   ", n)
        print(f"\nINSPECTED {inspected} archives: members={totals['members']} exact={totals['exact']} "
              f"last2={totals['last2']} name_only={totals['name_only']} wanted={wanted}")
        return 0
    if a.list or not (a.all or a.only):
        return 0
    if not a.confirm_reviewed_rights:
        raise SystemExit("refusing dataset acquisition without --confirm-reviewed-rights; verify release and media terms first")
    selected = [item for item in files if a.all or any(o.lower() in item[0].lower() for o in a.only)]
    if not selected:
        raise SystemExit("no archive members matched the requested selection")
    if a.stop_after or a.max_archives:
        selected.sort(key=lambda item: item[2])
    os.makedirs(a.out, exist_ok=True)
    required = required_free_bytes(selected, a.delete_zips)
    if shutil.disk_usage(a.out).free < required:
        raise SystemExit(f"insufficient free space (need about {required / 1e9:.1f} GB"
                         f"{' for the largest archive' if a.delete_zips else ' for all selected archives'} plus overhead)")
    digest = keep_digest(keep)
    failed, total, attempts = [], 0, 0

    def smoke_verdict():
        if a.stop_after and total >= a.stop_after:
            print(f"STOPPED EARLY: {total} videos extracted (--stop-after {a.stop_after}) from {attempts} archive(s)")
            return 0
        if a.stop_after and a.max_archives and attempts >= a.max_archives:
            print(f"GAVE UP: {attempts} archives tried and only {total} videos matched the keep list; "
                  "check the archive member names printed above or run --inspect all")
            return 2
        return None

    for k, url, size, checksum in selected:
        dst = os.path.join(a.out, k); done = dst + ".done"
        marker = f"{size}|{checksum}|{digest}\n"
        marker_ok = os.path.exists(done) and Path(done).read_text(encoding="utf-8") == marker
        is_zip = k.lower().endswith(".zip")
        if marker_ok and (a.delete_zips or a.remote) and is_zip and not os.path.exists(dst):
            print("already done (no archive kept)", k); continue
        if (marker_ok and os.path.exists(dst) and (not size or os.path.getsize(dst) == size) and
                _checksum_ok(dst, checksum)):
            print("already done", k); continue
        if a.remote and is_zip:
            try:
                rf = RangeFile(url, size, block=1 << 20)
                n = safe_extract_zip(rf, a.out, keep, a.max_extract_bytes)
            except (RangeNotSupported, OSError, ValueError, zipfile.BadZipFile, ManifestError) as exc:
                print(f"  remote read failed ({type(exc).__name__}: {exc}); falling back to a full download", flush=True)
            else:
                print(f"  extracted {n} files (fetched {rf.bytes_fetched / 1e6:.1f} MB in {rf.requests} requests)", flush=True)
                total += n; attempts += 1
                if keep is not None and n == 0:
                    print("  note: this archive holds none of the listed videos", flush=True)
                Path(done).write_text(marker, encoding="utf-8")
                verdict = smoke_verdict()
                if verdict is not None:
                    return verdict
                continue
        if (not os.path.exists(dst) or (size and os.path.getsize(dst) != size) or
                not _checksum_ok(dst, checksum)):
            print("downloading", k, flush=True)
            if not fetch(url, dst, size, checksum):
                print("  GAVE UP on", k, "- rerun this cell later", flush=True); failed.append(k); continue
        if is_zip:
            print("unzipping", k, flush=True)
            try:
                n = safe_extract_zip(dst, a.out, keep, a.max_extract_bytes)
            except (OSError, ValueError, zipfile.BadZipFile, ManifestError) as exc:
                print(f"  extraction failed safely: {exc}", flush=True)
                failed.append(k); continue
            print("  extracted", n, "files", flush=True); total += n; attempts += 1
            if keep is not None and n == 0:
                print("  WARNING: no archive member matched the --keep list for this part (fine if this archive holds none of the listed videos; use --inspect to check the layout)", flush=True)
                try:
                    count, sample = describe_zip(dst)
                    print(f"  archive holds {count} files; first names: {sample}", flush=True)
                except (OSError, zipfile.BadZipFile) as exc:
                    print(f"  could not list archive: {exc}", flush=True)
        Path(done).write_text(marker, encoding="utf-8")
        if a.delete_zips and is_zip:
            os.remove(dst)
        verdict = smoke_verdict()
        if verdict is not None:
            return verdict
    if failed:
        print("NOT FINISHED:", failed, "-> RERUN THIS CELL")
        return 1
    print("ALL FINISHED ->", a.out, "(videos extracted this run:", total, ")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
