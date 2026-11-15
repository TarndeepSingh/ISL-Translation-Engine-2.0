"""Idempotent, resumable orchestration of the INCLUDE-50 Colab workflow.

Colab gives a brand-new, empty virtual machine every session; only Google Drive survives. This module
records finished work on Drive and, each time it is called, does the least work needed:

* anything already finished is skipped (a ``state.json`` on Drive + a check that the files still exist);
* anything finished on a *previous* VM is copied back from Drive instead of recomputed;
* the big, slow inputs (videos) are only fetched if a later stage really needs them. Once landmarks
  exist on Drive, training on a fresh VM needs only ~tens of MB, not the 13 GB of videos.

Only the standard library is used, so it can be imported before the project environment is installed.
Every external command goes through ``ctx.runner`` so the whole flow can be tested with a fake.
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

POLICIES = {"official": "official", "session": "session-proxy", "balanced": "balanced"}     # short name -> include_manifest value
VARIANTS = {"baseline": [], "aug7-warp0.3": ["--aug-rotation-deg", "7", "--aug-time-warp", "0.3"]}
BATCH = 6                                                           # archives per download batch
PREFLIGHT_MIN_FRACTION = 0.95


class FlowError(RuntimeError):
    """A stage cannot continue; the message says what to do. Finished stages stay finished."""


# --------------------------------------------------------------------------------------------- runner

def default_runner(cmd, cwd, env, echo=True):
    """Run a command, stream its output live, and return (returncode, tail_of_output)."""
    process = subprocess.Popen(cmd, cwd=str(cwd), env={**os.environ, **env}, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines: list[str] = []
    for line in process.stdout:                       # type: ignore[union-attr]
        if echo:
            print(line, end="", flush=True)
        lines.append(line)
    process.wait()
    return process.returncode, "".join(lines[-400:])


# ------------------------------------------------------------------------------------------- settings

@dataclass
class Settings:
    rights_reviewed: bool = False
    policies: tuple[str, ...] = ("official",)
    seeds: tuple[int, ...] = (13, 14, 15)
    variants: tuple[str, ...] = ("baseline", "aug7-warp0.3")
    epochs: int = 150
    batch_size: int = 32
    allow_missing_videos: int = 0
    max_failure_fraction: float = 0.05
    sync_seconds: float = 120.0          # how often long jobs copy their progress to Drive

    def __post_init__(self):
        for policy in self.policies:
            if policy not in POLICIES:
                raise FlowError(f"unknown split policy {policy!r}; use one of {sorted(POLICIES)}")
        for variant in self.variants:
            if variant not in VARIANTS:
                raise FlowError(f"unknown training variant {variant!r}; use one of {sorted(VARIANTS)}")


class State:
    """Tiny JSON progress record kept on Drive (written atomically)."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def _read(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def get(self, key: str):
        return self._read().get(key)

    def set(self, key: str, **info) -> None:
        data = self._read()
        data[key] = {"done_at": time.strftime("%Y-%m-%d %H:%M:%S"), **info}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.path)

    def clear(self, key: str) -> bool:
        data = self._read()
        if key not in data:
            return False
        del data[key]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        return True

    def all(self) -> dict:
        return self._read()


@dataclass
class Ctx:
    project: Path
    drive: Path
    settings: Settings = field(default_factory=Settings)
    data: Path = Path("/content/isl-data")
    runner: Callable = default_runner
    log: Callable = print

    def __post_init__(self):
        self.project, self.drive, self.data = Path(self.project), Path(self.drive), Path(self.data)
        self.state = State(self.drive / "state" / "state.json")

    # local (fast, per-VM) locations
    @property
    def raw(self): return self.data / "raw"
    @property
    def keep(self): return self.data / "keep.txt"
    @property
    def manifests(self): return self.data / "manifests"
    @property
    def reports(self): return self.data / "reports"
    def proc(self, policy): return self.data / "processed" / f"include50-{policy}"
    def hashed(self, policy): return self.manifests / f"include50-{policy}-hashed.csv"
    # durable (Drive) locations
    @property
    def drive_raw(self): return self.drive / "raw_include50"
    @property
    def drive_source(self): return self.drive / "source"
    def drive_proc(self, policy): return self.drive / "processed" / f"include50-{policy}"
    def run_dir(self, policy, variant, seed):
        return self.drive / "runs" / "isolated" / f"include50-{policy}-{variant}-seed{seed}"


# ------------------------------------------------------------------------------------------ utilities

def py(ctx: Ctx, module: str, *args, echo: bool = True):
    cmd = ["uv", "run", "python", "-m", module, *[str(a) for a in args]]
    return ctx.runner(cmd, ctx.project, {"MPLBACKEND": "Agg"}, echo)


def copy_tree(src, dst, log=print, label: str = "", exclude: str | None = None) -> tuple[int, int]:
    """Incremental copy: files whose size matches and that are not older are skipped. Returns (copied, skipped)."""
    src, dst = Path(src), Path(dst)
    copied = skipped = 0
    if not src.is_dir():
        return 0, 0
    for item in sorted(src.rglob("*")):
        if not item.is_file() or item.name.endswith(".part") or (exclude and re.match(exclude, item.name)):
            continue
        target = dst / item.relative_to(src)
        if target.is_file() and target.stat().st_size == item.stat().st_size \
                and target.stat().st_mtime >= item.stat().st_mtime - 2:
            skipped += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + ".part")
        shutil.copy2(item, part)
        os.replace(part, target)
        copied += 1
        if copied % 200 == 0:
            log(f"  {label or src.name}: copied {copied} files ...")
    if copied:
        log(f"  {label or src.name}: copied {copied} new files, {skipped} already there")
    return copied, skipped


TEMP_CACHE = r"^tmp[a-z0-9_]{8}\.npz$"      # half-written landmark files (see ml.datasets.preprocess)


class periodic_sync:
    """While a long job runs, copy its output folder to Drive every ``interval`` seconds (and once at the end)."""

    def __init__(self, src, dst, interval: float, log=print, label: str = ""):
        self.src, self.dst, self.interval, self.log, self.label = src, dst, interval, log, label
        self._stop = threading.Event()
        self._thread = None

    def _loop(self):
        while not self._stop.wait(self.interval):
            copy_tree(self.src, self.dst, self.log, self.label, exclude=TEMP_CACHE)

    def __enter__(self):
        if self.interval and self.interval > 0:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30)
        copy_tree(self.src, self.dst, self.log, self.label, exclude=TEMP_CACHE)
        return False


def copy_file(src, dst) -> bool:
    src, dst = Path(src), Path(dst)
    if not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def read_keep(path) -> list[str]:
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def missing_files(root, entries) -> list[str]:
    root = Path(root)
    return [e for e in entries if ".." not in e.split("/") and not (root / e).is_file()]


def _require_rights(ctx: Ctx, what: str) -> None:
    if not ctx.settings.rights_reviewed:
        raise FlowError(f"To {what} you must first read the INCLUDE terms (https://zenodo.org/records/4010759) "
                        "and tick RIGHTS_REVIEWED in the settings cell, then run this cell again.")
    if not ctx.state.get("rights_reviewed"):
        ctx.state.set("rights_reviewed", note="confirmed by the user in the settings cell")


def _fail(ctx: Ctx, stage: str, code: int, tail: str, hint: str) -> None:
    ctx.log(f"\nSTAGE FAILED: {stage} (exit code {code}).")
    raise FlowError(f"{stage} failed (exit code {code}). {hint}\nLast output:\n{tail[-1500:]}")


# --------------------------------------------------------------------------------------------- stages

def audit(ctx: Ctx) -> int:
    """Fetch the INCLUDE metadata table once and keep the INCLUDE-50 video list; later days restore it."""
    if not ctx.keep.is_file():
        d_keep, d_rep = ctx.drive_source / "keep.txt", ctx.drive_source / "include_audit.json"
        if d_keep.is_file():
            ctx.log("audit: restoring the video list from Drive")
            copy_file(d_keep, ctx.keep)
            copy_file(d_rep, ctx.reports / "include_audit.json")
        else:
            ctx.log("audit: reading the INCLUDE metadata table (first time only)")
            ctx.reports.mkdir(parents=True, exist_ok=True)
            code, tail = py(ctx, "ml.datasets.include_manifest", "--write-keep", ctx.keep,
                            "--report", ctx.reports / "include_audit.json")
            if code != 0 or not ctx.keep.is_file():
                _fail(ctx, "audit", code, tail, "Send me this message; the Hugging Face table could not be read.")
            copy_file(ctx.keep, d_keep)
            copy_file(ctx.reports / "include_audit.json", d_rep)
    entries = len(read_keep(ctx.keep))
    ctx.state.set("audit", entries=entries)
    ctx.log(f"audit: {entries} INCLUDE-50 videos on the list")
    return entries


def _archive_names(ctx: Ctx) -> list[str]:
    code, text = py(ctx, "isl.download_include", "--list", echo=False)
    if code != 0:
        _fail(ctx, "listing archives", code, text, "Zenodo could not be reached.")
    names = re.findall(r"^\s*[\d.]+ GB\s+(\S+\.zip)\s*$", text, flags=re.MULTILINE)
    if not names:
        _fail(ctx, "listing archives", code, text, "No .zip archives were listed.")
    return names


def _preflight(ctx: Ctx, keep_path: Path) -> bool:
    """Read every archive's index with tiny range requests and confirm the layout matches the keep list.

    Returns True if range reads work (so ``--remote`` can be used); False to fall back to full downloads.
    Raises FlowError if the archives do not seem to contain the wanted videos.
    """
    ctx.log("preflight: checking archive layout (reads ~1 MB per archive, downloads no videos)")
    code, text = py(ctx, "isl.download_include", "--inspect", "all", "--keep", keep_path)
    summary = re.search(r"INSPECTED (\d+) archives: members=(\d+) exact=(\d+) last2=(\d+) name_only=(\d+) wanted=(\d+)", text)
    if code != 0 or not summary:
        ctx.log("preflight: range reads are not available here; will download whole archives instead")
        return False
    _, members, exact, last2, name_only, wanted = (int(x) for x in summary.groups())
    matched = max(exact, last2)
    ctx.log(f"preflight: {matched} of {wanted} wanted videos found (exact={exact}, label/file={last2}, name only={name_only})")
    if matched < PREFLIGHT_MIN_FRACTION * wanted:
        raise FlowError(f"Only {matched} of {wanted} wanted videos were found inside the archives (exact={exact}, "
                        f"label/file={last2}, file-name-only={name_only}). The archive folder layout differs from the "
                        "metadata. Nothing was downloaded. Send me the preflight output above.")
    return True


def ensure_videos(ctx: Ctx) -> None:
    """Make every INCLUDE-50 video exist under ``ctx.raw`` - from disk, Drive, or (resumably) from Zenodo."""
    keep = read_keep(ctx.keep)
    allowed = ctx.settings.allow_missing_videos
    if len(missing_files(ctx.raw, keep)) <= allowed:
        ctx.log("videos: already on this VM")
        return
    if ctx.drive_raw.is_dir():
        ctx.log("videos: restoring what is saved on Drive (includes progress markers)")
        copy_tree(ctx.drive_raw, ctx.raw, ctx.log, "videos")
        if len(missing_files(ctx.raw, keep)) <= allowed:
            ctx.state.set("videos", videos=len(keep) - len(missing_files(ctx.raw, keep)))
            return
    _require_rights(ctx, "download the INCLUDE videos")
    remote = _preflight(ctx, ctx.keep)
    base = ["--keep", ctx.keep, "--delete-zips", "--out", ctx.raw, "--confirm-reviewed-rights"]
    if remote:
        base.append("--remote")
    else:
        smoke = ctx.state.get("smoke")
        if not smoke:
            ctx.log("videos: smoke test - trying the smallest archives until some wanted videos appear")
            code, tail = py(ctx, "isl.download_include", "--all", "--stop-after", 5, "--max-archives", 4, *base)
            copy_tree(ctx.raw, ctx.drive_raw, ctx.log, "videos")
            if code != 0:
                _fail(ctx, "smoke test", code, tail, "No wanted video appeared in the smallest archives; send me this output.")
            ctx.state.set("smoke", ok=True)
    todo = [n for n in _archive_names(ctx) if not (ctx.raw / f"{n}.done").exists()]
    ctx.log(f"videos: {len(todo)} archives to process")
    failures = []
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        ctx.log(f"videos: archives {i + 1}-{i + len(batch)} of {len(todo)}")
        code, tail = py(ctx, "isl.download_include", "--only", *batch, *base)
        copy_tree(ctx.raw, ctx.drive_raw, ctx.log, "videos")          # progress survives a dead VM
        if code != 0:
            failures.append((batch, code, tail))
            break
    if failures:
        _, code, tail = failures[0]
        _fail(ctx, "video download", code, tail, "Progress is saved on Drive. Run this cell again to continue.")
    still_missing = missing_files(ctx.raw, keep)
    if len(still_missing) > allowed:
        raise FlowError(f"{len(still_missing)} of {len(keep)} videos are still missing after all archives "
                        f"(first: {still_missing[:3]}). Send me this message; set ALLOW_MISSING_VIDEOS only if "
                        "you accept training without them.")
    ctx.state.set("videos", videos=len(keep) - len(still_missing), missing=len(still_missing))
    ctx.log(f"videos: done ({len(keep) - len(still_missing)} of {len(keep)} present)")


def checked(ctx: Ctx, policy: str) -> None:
    """Build the manifest for a split policy and verify every video (hashes, decoding, splits, rights)."""
    key = f"checked:{policy}"
    hashed = ctx.hashed(policy)
    names = [hashed.name, f"include50-{policy}.csv"]
    reports = [f"validation-include50-{policy}.json", f"near-duplicates-include50-{policy}.json",
               f"include_manifest_{policy}.json"]
    if ctx.state.get(key):
        if not hashed.is_file():
            for name in names:
                copy_file(ctx.drive.joinpath("manifests", name), ctx.manifests / name)
            for name in reports:
                copy_file(ctx.drive.joinpath("reports", name), ctx.reports / name)
        if hashed.is_file():
            ctx.log(f"checks[{policy}]: already done")
            return
    _require_rights(ctx, "approve the manifest rights status")
    ensure_videos(ctx)
    ctx.manifests.mkdir(parents=True, exist_ok=True)
    ctx.reports.mkdir(parents=True, exist_ok=True)
    manifest = ctx.manifests / f"include50-{policy}.csv"
    steps = [
        ("manifest", "ml.datasets.include_manifest", "--data-root", ctx.raw, "--split-policy", POLICIES[policy],
         "--rights-status", "approved_research", "--out", manifest,
         "--report", ctx.reports / f"include_manifest_{policy}.json"),
        ("fingerprint", "ml.datasets.fingerprint", "--manifest", manifest, "--data-root", ctx.raw, "--out", hashed),
        ("validation", "ml.datasets.validate", "--manifest", hashed, "--data-root", ctx.raw, "--verify-hashes",
         "--decode-media", "--require-approved-rights", "--require-splits", "--require-group-ids",
         "--out", ctx.reports / f"validation-include50-{policy}.json"),
        ("near-duplicates", "ml.datasets.near_duplicates", "--manifest", hashed, "--data-root", ctx.raw,
         "--out", ctx.reports / f"near-duplicates-include50-{policy}.json"),
    ]
    for label, module, *args in steps:
        code, tail = py(ctx, module, *args)
        if code != 0:
            copy_tree(ctx.reports, ctx.drive / "reports", ctx.log, "reports")
            _fail(ctx, f"{label} ({policy})", code, tail,
                  f"Read the report in {ctx.reports} (copied to Drive/reports). Nothing after this step ran.")
    copy_tree(ctx.manifests, ctx.drive / "manifests", ctx.log, "manifests")
    copy_tree(ctx.reports, ctx.drive / "reports", ctx.log, "reports")
    ctx.state.set(key)
    ctx.log(f"checks[{policy}]: done")


def landmarks(ctx: Ctx, policy: str) -> Path:
    """Landmark features for a split policy; restored from Drive on later days (no videos needed)."""
    key = f"landmarks:{policy}"
    proc, index = ctx.proc(policy), ctx.proc(policy) / "index.csv"
    if ctx.state.get(key):
        if not index.is_file():
            ctx.log(f"landmarks[{policy}]: restoring from Drive")
            copy_tree(ctx.drive_proc(policy), proc, ctx.log, "landmarks")
        if index.is_file():
            ctx.log(f"landmarks[{policy}]: ready ({ctx.state.get(key).get('artifacts', '?')} clips)")
            return index
    checked(ctx, policy)
    ensure_videos(ctx)
    proc.mkdir(parents=True, exist_ok=True)
    if ctx.drive_proc(policy).is_dir():                  # a previous VM may have finished part of the work
        ctx.log(f"landmarks[{policy}]: restoring partial progress from Drive")
        copy_tree(ctx.drive_proc(policy), proc, ctx.log, "landmarks")
    with periodic_sync(proc, ctx.drive_proc(policy), ctx.settings.sync_seconds, ctx.log, "landmarks"):
        code, tail = py(ctx, "ml.datasets.preprocess", "--manifest", ctx.hashed(policy), "--data-root", ctx.raw,
                        "--task", "isolated", "--out", proc, "--continue-on-error")
    if code != 0 or not index.is_file():
        _fail(ctx, f"landmark extraction ({policy})", code, tail,
              "Finished videos are kept; run this cell again to continue where it stopped.")
    with index.open(newline="", encoding="utf-8") as stream:
        artifacts = sum(1 for _ in csv.DictReader(stream))
    failures = 0
    report = proc / "preprocess_failures.json"
    if report.is_file():
        failures = int(json.loads(report.read_text(encoding="utf-8")).get("failed", 0))
    attempted = artifacts + failures
    if attempted and failures / attempted > ctx.settings.max_failure_fraction:
        copy_tree(proc, ctx.drive_proc(policy), ctx.log, "landmarks")
        raise FlowError(f"{failures} of {attempted} videos failed landmark extraction (more than "
                        f"{ctx.settings.max_failure_fraction:.0%}). See {report}. Send me that file.")
    copy_tree(proc, ctx.drive_proc(policy), ctx.log, "landmarks")
    ctx.state.set(key, artifacts=artifacts, failed=failures)
    ctx.log(f"landmarks[{policy}]: done ({artifacts} clips, {failures} failed)")
    return index


def prepare(ctx: Ctx) -> None:
    """Everything up to 'ready to train', skipping whatever is already done."""
    audit(ctx)
    for policy in ctx.settings.policies:
        landmarks(ctx, policy)
    ctx.log("\nPREPARE COMPLETE - data is ready; continue with the training cell.")


def train_all(ctx: Ctx) -> list[Path]:
    """Train every (variant, seed) not already finished. Runs live on Drive; finished runs are skipped."""
    done: list[Path] = []
    for policy in ctx.settings.policies:
        index = landmarks(ctx, policy)
        for variant in ctx.settings.variants:
            for seed in ctx.settings.seeds:
                run = ctx.run_dir(policy, variant, seed)
                if (run / "experiment.json").is_file():
                    ctx.log(f"train[{policy}/{variant}/seed{seed}]: already finished")
                    done.append(run)
                    continue
                ctx.log(f"train[{policy}/{variant}/seed{seed}]: starting")
                run.parent.mkdir(parents=True, exist_ok=True)
                code, tail = py(ctx, "isl.train", "--index", index, "--out", run, "--epochs", ctx.settings.epochs,
                                "--bs", ctx.settings.batch_size, "--seed", seed, *VARIANTS[variant])
                if code != 0 or not (run / "experiment.json").is_file():
                    _fail(ctx, f"training {policy}/{variant}/seed{seed}", code, tail,
                          "Finished runs are kept; run this cell again to retry only this one.")
                done.append(run)
    return done


def _test_clips(ctx: Ctx, policy: str) -> int:
    index = ctx.proc(policy) / "index.csv"
    if not index.is_file():
        return 0
    with index.open(newline="", encoding="utf-8") as stream:
        return sum(1 for row in csv.DictReader(stream) if row.get("split") == "test")


def summary(ctx: Ctx, policy: str | None = None) -> list[dict]:
    """Compare finished runs: mean +/- spread over seeds, with the 95% sampling interval of the test set."""
    rows = []
    for pol in ([policy] if policy else list(ctx.settings.policies)):
        n_test = _test_clips(ctx, pol)
        for variant in ctx.settings.variants:
            top1, top3 = [], []
            for seed in ctx.settings.seeds:
                path = ctx.run_dir(pol, variant, seed) / "experiment.json"
                if path.is_file():
                    data = json.loads(path.read_text(encoding="utf-8"))
                    top1.append(float(data["test_top1"]))
                    top3.append(float(data["test_top3"]))
            if not top1:
                continue
            mean = statistics.fmean(top1)
            rows.append({
                "policy": pol, "variant": variant, "seeds": len(top1), "top1": top1,
                "top1_mean": mean, "top1_std": statistics.stdev(top1) if len(top1) > 1 else 0.0,
                "top3_mean": statistics.fmean(top3), "test_clips": n_test,
                "interval95": 1.96 * math.sqrt(mean * (1 - mean) / n_test) if n_test else float("nan"),
            })
    return rows


def format_summary(rows: list[dict]) -> str:
    if not rows:
        return "No finished runs yet."
    head = f"{'split':9} {'variant':14} {'seeds':>5} {'top-1 mean':>11} {'+/- seed std':>13} {'top-3 mean':>11} {'95% test interval':>19}"
    lines = [head, "-" * len(head)]
    for r in rows:
        interval = f"+/-{100 * r['interval95']:.1f} pts" if r["test_clips"] else "n/a"
        lines.append(f"{r['policy']:9} {r['variant']:14} {r['seeds']:>5} {100 * r['top1_mean']:>10.1f}% "
                     f"{100 * r['top1_std']:>11.1f}pt {100 * r['top3_mean']:>10.1f}% {interval:>19}")
        lines.append(f"{'':9} per-seed top-1: {', '.join(f'{100 * v:.1f}%' for v in r['top1'])}  "
                     f"(test clips: {r['test_clips']})")
    lines.append("A difference between variants is only meaningful if it is larger than the seed spread AND the interval.")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------- inspection

STAGES = ("audit", "videos", "smoke", "checked:official", "checked:session", "landmarks:official", "landmarks:session")


def show_state(ctx: Ctx) -> str:
    data = ctx.state.all()
    lines = ["Saved progress (on Drive):"]
    for key in sorted(data):
        info = {k: v for k, v in data[key].items() if k != "done_at"}
        lines.append(f"  {key:22} done {data[key].get('done_at', '?')}  {info if info else ''}")
    if len(lines) == 1:
        lines.append("  (nothing finished yet)")
    return "\n".join(lines)


def reset(ctx: Ctx, *stages: str) -> None:
    """Forget finished stages so they run again next time (does not delete any files)."""
    for stage in stages:
        ctx.log(f"reset {stage}: {'cleared' if ctx.state.clear(stage) else 'was not recorded'}")
