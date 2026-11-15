"""Tests for ``isl.flow``: the idempotent, resumable Colab workflow.

A simulated world stands in for Colab: each 'VM' is a fresh local data directory, 'Drive' is a directory
that survives, and every external command is a fake that writes plausible placeholder files. The tests
check WHICH commands run on day 1, day 2, after failures and after interruptions - i.e. that finished
work is not repeated and that nothing heavy is fetched when it is not needed. They say nothing about
the quality of any real model or video.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

import isl.flow as flow


class World:
    def __init__(self, n=12, archives=("A_1of2.zip", "A_2of2.zip", "B_1of2.zip", "B_2of2.zip")):
        self.entries = [f"Animals/{i}. Dog/MVI_{i:04d}.MOV" for i in range(n)]
        self.archives = list(archives)
        self.calls: list[tuple[str, list[str]]] = []
        self.inspect = "ok"                    # ok | unavailable | mismatch
        self.validate_rc = 0
        self.preprocess_partial = False
        self.preprocess_failed = 0
        self.fail_download_batch = None        # 1-based non-smoke batch number that fails
        self.missing_video: set[str] = set()
        self.downloaded: list[str] = []
        self.preprocess_existing_at_start = 0
        self._batches = 0

    # ---- helpers
    def modules(self):
        return [m for m, _ in self.calls]

    def count(self, module, flag=None):
        return sum(1 for m, a in self.calls if m == module and (flag is None or flag in a))

    @staticmethod
    def opt(args, name):
        return args[args.index(name) + 1]

    def runner(self, cmd, cwd, env, echo=True):
        module, args = cmd[4], [str(a) for a in cmd[5:]]
        self.calls.append((module, args))
        return getattr(self, "_" + module.replace(".", "_"))(args)

    # ---- fake commands
    def _ml_datasets_include_manifest(self, args):
        if "--write-keep" in args:
            Path(self.opt(args, "--write-keep")).parent.mkdir(parents=True, exist_ok=True)
            Path(self.opt(args, "--write-keep")).write_text("\n".join(self.entries) + "\n")
        elif "--out" in args:
            Path(self.opt(args, "--out")).parent.mkdir(parents=True, exist_ok=True)
            Path(self.opt(args, "--out")).write_text("manifest")
        if "--report" in args:
            Path(self.opt(args, "--report")).parent.mkdir(parents=True, exist_ok=True)
            Path(self.opt(args, "--report")).write_text("{}")
        return 0, ""

    def _isl_download_include(self, args):
        n = len(self.entries)
        if "--list" in args:
            lines = [f"  {1.0 + i / 10:6.2f} GB  {name}" for i, name in enumerate(self.archives)]
            return 0, "\n".join(lines + ["  0.00 GB  README.md", "total 4.4 GB in 5 files"])
        if "--inspect" in args:
            if self.inspect == "unavailable":
                return 3, "INSPECT UNAVAILABLE for A_1of2.zip: RangeNotSupported: HTTP 200"
            exact, last2 = (n, n) if self.inspect == "ok" else (0, 1)
            return 0, f"INSPECTED 4 archives: members=48 exact={exact} last2={last2} name_only=0 wanted={n}"
        smoke = "--stop-after" in args
        if "--only" in args:
            start = args.index("--only") + 1
            names = []
            while start < len(args) and not args[start].startswith("--"):
                names.append(args[start]); start += 1
        else:
            names = list(self.archives)
        if smoke:
            names = names[:1]
        else:
            self._batches += 1
            if self.fail_download_batch == self._batches:
                return 1, "NOT FINISHED: ['x'] -> RERUN THIS CELL"
        out = Path(self.opt(args, "--out"))
        for name in names:
            self.downloaded.append(name)
            idx = self.archives.index(name)
            for entry in self.entries[idx::len(self.archives)]:
                if entry in self.missing_video:
                    continue
                target = out / entry
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"video")
            (out / f"{name}.done").write_text("done")
        return 0, "STOPPED EARLY" if smoke else "ALL FINISHED"

    def _ml_datasets_fingerprint(self, args):
        Path(self.opt(args, "--out")).parent.mkdir(parents=True, exist_ok=True)
        Path(self.opt(args, "--out")).write_text("hashed")
        return 0, ""

    def _ml_datasets_validate(self, args):
        Path(self.opt(args, "--out")).parent.mkdir(parents=True, exist_ok=True)
        Path(self.opt(args, "--out")).write_text(json.dumps({"ok": self.validate_rc == 0}))
        return self.validate_rc, "validation output"

    def _ml_datasets_near_duplicates(self, args):
        Path(self.opt(args, "--out")).write_text("{}")
        return 0, ""

    def _ml_datasets_preprocess(self, args):
        out = Path(self.opt(args, "--out"))
        out.mkdir(parents=True, exist_ok=True)
        self.preprocess_existing_at_start = len(list(out.glob("clip_*.npz")))
        n = len(self.entries)
        todo = range(n // 2) if self.preprocess_partial else range(n - self.preprocess_failed)
        for i in todo:
            (out / f"clip_{i}.npz").write_bytes(b"features")
        if self.preprocess_partial:
            return 1, "interrupted"
        splits = ["train"] * (n - 4) + ["val"] * 2 + ["test"] * 2
        rows = ["sample_id,split,cache_file,cache"] + [
            f"s{i},{splits[i]},clip_{i}.npz,/old/vm/clip_{i}.npz" for i in range(n - self.preprocess_failed)]
        (out / "index.csv").write_text("\n".join(rows) + "\n")
        if self.preprocess_failed:
            (out / "preprocess_failures.json").write_text(json.dumps({"failed": self.preprocess_failed}))
        return 0, ""

    def _isl_train(self, args):
        run = Path(self.opt(args, "--out"))
        run.mkdir(parents=True, exist_ok=True)
        seed, aug = int(self.opt(args, "--seed")), "--aug-rotation-deg" in args
        (run / "experiment.json").write_text(json.dumps(
            {"test_top1": (0.90 if aug else 0.88) + (seed - 13) * 0.01, "test_top3": 0.97}))
        return 0, ""


def make_ctx(tmp_path, world, vm="vm1", rights=True, **kwargs):
    project = tmp_path / "ISL"
    project.mkdir(parents=True, exist_ok=True)
    settings = flow.Settings(rights_reviewed=rights, policies=("official",), seeds=(13,), variants=("baseline",),
                             sync_seconds=0, **kwargs)
    return flow.Ctx(project=project, drive=tmp_path / "drive", settings=settings, data=tmp_path / vm / "data",
                    runner=world.runner, log=lambda *a, **k: None)


# ------------------------------------------------------------------------------------------ day 1 / day 2

def test_first_day_runs_every_stage_in_order_and_saves_to_drive(tmp_path):
    world = World()
    ctx = make_ctx(tmp_path, world)
    flow.prepare(ctx)
    order = [(m, a) for m, a in world.calls]
    seq = [m for m, _ in order]
    first = {m: seq.index(m) for m in set(seq)}
    assert first["ml.datasets.include_manifest"] < first["isl.download_include"] < first["ml.datasets.fingerprint"] \
        < first["ml.datasets.validate"] < first["ml.datasets.near_duplicates"] < first["ml.datasets.preprocess"]
    drive = tmp_path / "drive"
    assert (drive / "source" / "keep.txt").is_file()
    assert all((drive / "raw_include50" / e).is_file() for e in world.entries)
    assert (drive / "manifests" / "include50-official-hashed.csv").is_file()
    assert (drive / "processed" / "include50-official" / "index.csv").is_file()
    saved = ctx.state.all()
    assert {"audit", "videos", "checked:official", "landmarks:official"} <= set(saved)
    assert flow.train_all(ctx) == [ctx.run_dir("official", "baseline", 13)]


def test_second_day_new_vm_restores_from_drive_and_never_touches_videos(tmp_path):
    world = World()
    flow.prepare(make_ctx(tmp_path, world))
    flow.train_all(make_ctx(tmp_path, world))
    world.calls.clear()
    day2 = make_ctx(tmp_path, world, vm="vm2", rights=False)          # fresh VM, rights box NOT ticked
    flow.prepare(day2)
    flow.train_all(day2)
    assert world.calls == []                                           # no command at all ran
    assert (day2.proc("official") / "index.csv").is_file()             # landmarks restored from Drive
    assert not list(day2.raw.rglob("*.MOV")) if day2.raw.exists() else True   # 13 GB of videos not copied back


def test_rights_gate_blocks_downloads(tmp_path):
    world = World()
    with pytest.raises(flow.FlowError, match="RIGHTS_REVIEWED"):
        flow.prepare(make_ctx(tmp_path, world, rights=False))
    assert world.count("isl.download_include") == 0


# ----------------------------------------------------------------------------------------- video stage

def test_layout_mismatch_stops_before_downloading_anything(tmp_path):
    world = World()
    world.inspect = "mismatch"
    with pytest.raises(flow.FlowError, match="Nothing was downloaded"):
        flow.prepare(make_ctx(tmp_path, world))
    assert world.downloaded == []


def test_range_reads_unavailable_falls_back_to_bounded_smoke_test_then_full_download(tmp_path):
    world = World()
    world.inspect = "unavailable"
    ctx = make_ctx(tmp_path, world)
    flow.prepare(ctx)
    downloads = [a for m, a in world.calls if m == "isl.download_include" and "--list" not in a and "--inspect" not in a]
    assert "--stop-after" in downloads[0] and "--max-archives" in downloads[0]
    assert all("--remote" not in a for a in downloads)
    assert ctx.state.get("smoke")


def test_remote_mode_is_used_when_range_reads_work(tmp_path):
    world = World()
    flow.prepare(make_ctx(tmp_path, world))
    downloads = [a for m, a in world.calls if m == "isl.download_include" and "--list" not in a and "--inspect" not in a]
    assert downloads and all("--remote" in a for a in downloads)


def test_interrupted_download_resumes_with_only_the_remaining_archives(tmp_path, monkeypatch):
    monkeypatch.setattr(flow, "BATCH", 2)
    world = World()
    world.fail_download_batch = 2                                      # first batch ok, second fails
    with pytest.raises(flow.FlowError, match="Run this cell again"):
        flow.prepare(make_ctx(tmp_path, world))
    assert world.downloaded == ["A_1of2.zip", "A_2of2.zip", "B_1of2.zip", "B_2of2.zip"][:2]
    assert (tmp_path / "drive" / "raw_include50" / "A_1of2.zip.done").is_file()     # progress saved on Drive
    world.fail_download_batch, world.downloaded = None, []
    ctx2 = make_ctx(tmp_path, world, vm="vm2")                          # new VM
    flow.prepare(ctx2)
    assert world.downloaded == ["B_1of2.zip", "B_2of2.zip"]             # only what was left
    assert not flow.missing_files(ctx2.raw, flow.read_keep(ctx2.keep))


def test_videos_missing_from_archives_are_an_error_unless_explicitly_allowed(tmp_path):
    world = World()
    world.missing_video = {world.entries[0]}
    with pytest.raises(flow.FlowError, match="1 of 12 videos are still missing"):
        flow.prepare(make_ctx(tmp_path, world))
    world2 = World()
    world2.missing_video = {world2.entries[0]}
    flow.prepare(make_ctx(tmp_path / "ok", world2, allow_missing_videos=1))


# ---------------------------------------------------------------------------- validation and landmarks

def test_validation_failure_stops_before_landmarks_and_is_retried_after_fixing(tmp_path):
    world = World()
    world.validate_rc = 1
    ctx = make_ctx(tmp_path, world)
    with pytest.raises(flow.FlowError, match="validation"):
        flow.prepare(ctx)
    assert world.count("ml.datasets.preprocess") == 0
    assert ctx.state.get("checked:official") is None
    assert (tmp_path / "drive" / "reports" / "validation-include50-official.json").is_file()
    world.validate_rc = 0
    world.calls.clear()
    flow.prepare(ctx)
    assert world.count("isl.download_include") == 0                     # videos were already there
    assert world.count("ml.datasets.preprocess") == 1


def test_dead_vm_during_landmarks_keeps_partial_progress_and_resumes(tmp_path):
    world = World()
    world.preprocess_partial = True
    with pytest.raises(flow.FlowError, match="landmark extraction"):
        flow.prepare(make_ctx(tmp_path, world))
    assert len(list((tmp_path / "drive" / "processed" / "include50-official").glob("clip_*.npz"))) == 6
    world.preprocess_partial = False
    ctx2 = make_ctx(tmp_path, world, vm="vm2")
    flow.prepare(ctx2)
    assert world.preprocess_existing_at_start == 6                      # partial work restored BEFORE the rerun
    assert ctx2.state.get("landmarks:official")["artifacts"] == 12


def test_too_many_failed_videos_is_an_error_but_a_few_are_tolerated(tmp_path):
    world = World()
    world.preprocess_failed = 6                                         # 6 of 12 = 50%
    with pytest.raises(flow.FlowError, match="failed landmark extraction"):
        flow.prepare(make_ctx(tmp_path, world))
    world2 = World()
    world2.preprocess_failed = 1                                        # ~8% > 5% default -> still too many
    with pytest.raises(flow.FlowError):
        flow.prepare(make_ctx(tmp_path / "b", world2))
    world3 = World()
    world3.preprocess_failed = 1
    flow.prepare(make_ctx(tmp_path / "c", world3, max_failure_fraction=0.10))


# ---------------------------------------------------------------------------------------------- training

def test_training_runs_only_missing_seeds(tmp_path):
    world = World()
    ctx = make_ctx(tmp_path, world)
    flow.prepare(ctx)
    flow.train_all(ctx)
    assert world.count("isl.train") == 1
    ctx.settings.seeds = (13, 14)
    flow.train_all(ctx)
    assert world.count("isl.train") == 2                                # only seed 14 was added


def test_summary_reports_mean_spread_and_test_set_interval(tmp_path):
    world = World()
    ctx = make_ctx(tmp_path, world)
    ctx.settings.seeds, ctx.settings.variants = (13, 14, 15), ("baseline", "aug7-warp0.3")
    flow.prepare(ctx)
    flow.train_all(ctx)
    rows = {r["variant"]: r for r in flow.summary(ctx)}
    assert rows["baseline"]["top1"] == pytest.approx([0.88, 0.89, 0.90])
    assert rows["baseline"]["top1_mean"] == pytest.approx(0.89)
    assert rows["aug7-warp0.3"]["top1_mean"] == pytest.approx(0.91)
    assert rows["baseline"]["test_clips"] == 2
    assert rows["baseline"]["interval95"] == pytest.approx(1.96 * (0.89 * 0.11 / 2) ** 0.5)
    text = flow.format_summary(list(rows.values()))
    assert "baseline" in text and "89.0%" in text and "seed spread" in text


def test_summary_with_no_runs_says_so(tmp_path):
    assert flow.format_summary(flow.summary(make_ctx(tmp_path, World()))) == "No finished runs yet."


# ------------------------------------------------------------------------------------------- utilities

def test_state_show_and_reset(tmp_path):
    ctx = make_ctx(tmp_path, World())
    flow.prepare(ctx)
    assert "landmarks:official" in flow.show_state(ctx)
    flow.reset(ctx, "landmarks:official", "never-was")
    assert ctx.state.get("landmarks:official") is None
    world = ctx.runner.__self__
    world.calls.clear()
    flow.landmarks(ctx, "official")                                     # a reset stage runs again...
    assert world.count("ml.datasets.preprocess") == 1                   # ...(preprocessing itself resumes from its cache)
    assert ctx.state.get("landmarks:official") is not None              # ...and is recorded as done again


def test_state_writes_are_atomic_and_survive_garbage(tmp_path):
    state = flow.State(tmp_path / "s" / "state.json")
    state.set("a", n=1)
    assert state.get("a")["n"] == 1
    state.path.write_text("{not json")
    assert state.get("a") is None and state.all() == {}
    state.set("b")
    assert state.get("b") is not None and not list(state.path.parent.glob("*.tmp"))


def test_copy_tree_is_incremental_and_ignores_partial_and_excluded_files(tmp_path):
    src, dst = tmp_path / "s", tmp_path / "d"
    (src / "sub").mkdir(parents=True)
    (src / "a.txt").write_text("a")
    (src / "sub" / "b.txt").write_text("bb")
    (src / "c.part").write_text("junk")
    (src / "tmpab12cd34.npz").write_text("half")
    assert flow.copy_tree(src, dst, lambda *a: None, exclude=flow.TEMP_CACHE) == (2, 0)
    assert not (dst / "c.part").exists() and not (dst / "tmpab12cd34.npz").exists()
    assert flow.copy_tree(src, dst, lambda *a: None, exclude=flow.TEMP_CACHE) == (0, 2)
    time.sleep(0.01)
    (src / "a.txt").write_text("changed-and-longer")
    os.utime(src / "a.txt", (time.time() + 10, time.time() + 10))
    assert flow.copy_tree(src, dst, lambda *a: None, exclude=flow.TEMP_CACHE)[0] == 1
    assert (dst / "a.txt").read_text() == "changed-and-longer"


def test_periodic_sync_copies_while_the_job_is_still_running(tmp_path):
    src, dst = tmp_path / "s", tmp_path / "d"
    src.mkdir()
    with flow.periodic_sync(src, dst, 0.05, lambda *a: None):
        (src / "early.npz").write_text("x")
        deadline = time.time() + 3
        while not (dst / "early.npz").exists() and time.time() < deadline:
            time.sleep(0.02)
        assert (dst / "early.npz").exists()                             # copied BEFORE the job finished
        (src / "late.npz").write_text("y")
    assert (dst / "late.npz").exists()                                  # final sync on exit


def test_unknown_policy_or_variant_is_rejected():
    with pytest.raises(flow.FlowError, match="split policy"):
        flow.Settings(policies=("nonsense",))
    with pytest.raises(flow.FlowError, match="training variant"):
        flow.Settings(variants=("nonsense",))


# ---- contract tests: the flow parses the REAL downloader's output, not just my fake ----------------------

def _real_downloader_runner(monkeypatch, archives):
    """Runner that executes isl.download_include.main in-process against fake archives."""
    import contextlib
    import io

    import isl.download_include as dl

    listing = {"files": [{"key": k, "links": {"self": f"fake://{k}"}, "size": len(v), "checksum": ""}
                         for k, v in archives.items()]}
    listing["files"].append({"key": "README.md", "links": {"self": "fake://README.md"}, "size": 10, "checksum": ""})

    class Response(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *e): return False

    class FakeRange(io.BytesIO):
        def __init__(self, url, size=None, **kw):
            super().__init__(archives[url.removeprefix("fake://")])
            self.bytes_fetched, self.requests = 0, 1

    monkeypatch.setattr(dl.urllib.request, "urlopen", lambda *a, **k: Response(json.dumps(listing).encode()))
    monkeypatch.setattr(dl, "RangeFile", FakeRange)

    def runner(cmd, cwd, env, echo=True):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = dl.main([str(a) for a in cmd[5:]])
        return code, buffer.getvalue()
    return runner


def _zip_bytes(members):
    import io
    import zipfile
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        for name, data in members.items():
            bundle.writestr(name, data)
    return buffer.getvalue()


def test_flow_parses_the_real_downloader_archive_list(tmp_path, monkeypatch):
    archives = {"Animals_1of2.zip": _zip_bytes({"Animals/1. Dog/a.MOV": b"x"}),
                "Colours_1of1.zip": _zip_bytes({"Colours/2. Red/b.MOV": b"y"})}
    ctx = make_ctx(tmp_path, World())
    ctx.runner = _real_downloader_runner(monkeypatch, archives)
    assert sorted(flow._archive_names(ctx)) == ["Animals_1of2.zip", "Colours_1of1.zip"]   # README.md excluded


def test_flow_preflight_understands_the_real_inspect_summary(tmp_path, monkeypatch):
    archives = {"Animals_1of1.zip": _zip_bytes({"Animals/1. Dog/a.MOV": b"x", "Animals/1. Dog/other.MOV": b"z"}),
                "Colours_1of1.zip": _zip_bytes({"Colours/2. Red/b.MOV": b"y"})}
    ctx = make_ctx(tmp_path, World())
    ctx.runner = _real_downloader_runner(monkeypatch, archives)
    keep = tmp_path / "keep.txt"
    keep.write_text("Animals/1. Dog/a.MOV\nColours/2. Red/b.MOV\n")
    assert flow._preflight(ctx, keep) is True                          # 2 of 2 wanted found
    keep.write_text("\n".join(f"Missing/{i}. x/v{i}.MOV" for i in range(10)) + "\n")
    with pytest.raises(flow.FlowError, match="Nothing was downloaded"):
        flow._preflight(ctx, keep)                                     # 0 of 10 found -> blocks the download
