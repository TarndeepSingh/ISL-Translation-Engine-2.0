"""End-to-end tests of ``isl.download_include.main`` against a FAKE Zenodo record and fake archives.

No network is used: the record listing and ``fetch`` are replaced. Archive contents are placeholder bytes.
These cover defects the first real Colab run exposed: the size-total crash (it broke every invocation,
including --list), re-downloading finished archives after --delete-zips, and extracted files landing
somewhere other than the manifest's path.
"""
from __future__ import annotations

import io
import json
import zipfile

import pytest

import isl.download_include as dl

KEEP = ["Electronics/53. Fan/MVI_4532.MOV", "Animals/1. Dog/MVI_3030.MOV"]


def make_zip(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        for name, data in members.items():
            bundle.writestr(name, data)
    return buffer.getvalue()


ARCHIVES = {
    # extra top-level folder AND different capitalisation than the keep list / manifest
    "Electronics_2of2.zip": make_zip({"Electronics_2of2/electronics/53. fan/mvi_4532.mov": b"fan-video",
                                      "Electronics_2of2/electronics/53. fan/MVI_9999.MOV": b"not-wanted"}),
    "Animals_1of1.zip": make_zip({"Animals/1. Dog/MVI_3030.MOV": b"dog-video"}),
}


@pytest.fixture
def fake(monkeypatch, tmp_path):
    state = {"fetches": []}
    listing = {"files": [{"key": k, "links": {"self": f"fake://{k}"}, "size": len(v), "checksum": ""}
                         for k, v in ARCHIVES.items()]}

    class Response(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *exc): return False

    monkeypatch.setattr(dl.urllib.request, "urlopen", lambda url, *a, **k: Response(json.dumps(listing).encode()))

    def fake_fetch(url, dst, size, checksum="", tries=12):
        state["fetches"].append(url)
        if state.get("fail"):
            return False
        with open(dst, "wb") as stream:
            stream.write(ARCHIVES[url.removeprefix("fake://")])
        return True

    monkeypatch.setattr(dl, "fetch", fake_fetch)
    keep = tmp_path / "keep.txt"
    keep.write_text("\n".join(KEEP) + "\n", encoding="utf-8")
    state.update(keep=str(keep), out=str(tmp_path / "raw"), tmp=tmp_path)
    return state


def run(fake, *extra):
    return dl.main(["--out", fake["out"], "--keep", fake["keep"], *extra])


def test_list_prints_archives_and_a_numeric_total_without_rights_flag(fake, capsys):
    assert dl.main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "Electronics_2of2.zip" in out and "total " in out and "2 files" in out


def test_total_is_the_sum_of_sizes_not_checksums(fake, capsys, monkeypatch):
    big = [("a.zip", "u", 1_500_000_000, "md5:abc"), ("b.zip", "u", 2_500_000_000, "md5:def")]
    monkeypatch.setattr(dl, "list_archives", lambda api=None: big)
    dl.main(["--list"])
    assert "total 4.0 GB in 2 files" in capsys.readouterr().out


def test_download_is_refused_without_rights_confirmation(fake):
    with pytest.raises(SystemExit, match="confirm-reviewed-rights"):
        run(fake, "--all")
    assert fake["fetches"] == []


def test_unmatched_only_selection_is_an_error(fake):
    with pytest.raises(SystemExit, match="no archive members matched"):
        run(fake, "--only", "Nothing_1of1", "--confirm-reviewed-rights")


def test_full_run_extracts_at_manifest_paths_and_deletes_archives(fake, capsys):
    assert run(fake, "--all", "--delete-zips", "--confirm-reviewed-rights") == 0
    from pathlib import Path
    out = Path(fake["out"])
    # the archive's own prefix/case differ, yet files sit exactly at the keep-list (manifest) path
    assert (out / "Electronics/53. Fan/MVI_4532.MOV").read_bytes() == b"fan-video"
    assert (out / "Animals/1. Dog/MVI_3030.MOV").read_bytes() == b"dog-video"
    assert not (out / "Electronics/53. Fan/MVI_9999.MOV").exists()          # not in keep list
    assert not list(out.glob("*.zip"))
    assert "ALL FINISHED" in capsys.readouterr().out


def test_rerun_after_delete_zips_does_not_download_again(fake, capsys):
    run(fake, "--all", "--delete-zips", "--confirm-reviewed-rights")
    first = len(fake["fetches"])
    assert run(fake, "--all", "--delete-zips", "--confirm-reviewed-rights") == 0
    assert len(fake["fetches"]) == first == 2
    assert "already done (no archive kept)" in capsys.readouterr().out


def test_changed_keep_list_re_processes_finished_archives(fake):
    run(fake, "--all", "--delete-zips", "--confirm-reviewed-rights")
    with open(fake["keep"], "a", encoding="utf-8") as stream:
        stream.write("Animals/1. Dog/MVI_0000.MOV\n")
    run(fake, "--all", "--delete-zips", "--confirm-reviewed-rights")
    assert len(fake["fetches"]) == 4


def test_failed_download_is_reported_and_returns_nonzero(fake, capsys):
    fake["fail"] = True
    assert run(fake, "--all", "--delete-zips", "--confirm-reviewed-rights") == 1
    assert "NOT FINISHED" in capsys.readouterr().out


def test_zero_matching_members_warns(fake, capsys):
    from pathlib import Path
    Path(fake["keep"]).write_text("Weird/x/none.MOV\n", encoding="utf-8")
    run(fake, "--only", "Animals_1of1", "--delete-zips", "--confirm-reviewed-rights")
    assert "WARNING: no archive member matched" in capsys.readouterr().out


# ---- diagnostics, label/file fallback, and bounded smoke test -------------------------------------------

def test_zero_match_archive_reports_what_it_contains(fake, capsys):
    from pathlib import Path
    Path(fake["keep"]).write_text("Weird/x/none.MOV\n", encoding="utf-8")
    run(fake, "--only", "Animals_1of1", "--delete-zips", "--confirm-reviewed-rights")
    out = capsys.readouterr().out
    assert "archive holds 1 files" in out and "Animals/1. Dog/MVI_3030.MOV" in out


def test_label_file_fallback_when_archive_has_no_category_folder(fake):
    from pathlib import Path
    ARCHIVES["NoCategory_1of1.zip"] = make_zip({"53. fan/mvi_4532.mov": b"flat-fan"})
    fake_files = {"files": [{"key": "NoCategory_1of1.zip", "links": {"self": "fake://NoCategory_1of1.zip"},
                             "size": len(ARCHIVES["NoCategory_1of1.zip"]), "checksum": ""}]}
    import json as _json
    import isl.download_include as module

    class Resp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *e): return False
    module.urllib.request.urlopen = lambda url, *a, **k: Resp(_json.dumps(fake_files).encode())
    try:
        assert run(fake, "--all", "--delete-zips", "--confirm-reviewed-rights") == 0
    finally:
        del ARCHIVES["NoCategory_1of1.zip"]
    assert (Path(fake["out"]) / "Electronics/53. Fan/MVI_4532.MOV").read_bytes() == b"flat-fan"


def test_ambiguous_label_file_is_never_guessed(tmp_path):
    archive = tmp_path / "a.zip"
    archive.write_bytes(make_zip({"1. dog/mvi_1.mov": b"x"}))
    out = tmp_path / "o"
    out.mkdir()
    keep = ["Animals/1. Dog/MVI_1.MOV", "Pets/1. Dog/MVI_1.MOV"]     # same label/file under two categories
    assert dl.safe_extract_zip(str(archive), out, keep) == 0


def _two_archive_world(monkeypatch, fake, sizes_and_members):
    import json as _json
    listing = {"files": [{"key": k, "links": {"self": f"fake://{k}"}, "size": len(make_zip(m)) + extra, "checksum": ""}
                         for k, (m, extra) in sizes_and_members.items()]}

    class Resp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *e): return False
    monkeypatch.setattr(dl.urllib.request, "urlopen", lambda url, *a, **k: Resp(_json.dumps(listing).encode()))
    members = {k: make_zip(m) for k, (m, _) in sizes_and_members.items()}

    def fetch(url, dst, size, checksum="", tries=12):
        fake["fetches"].append(url.removeprefix("fake://"))
        open(dst, "wb").write(members[url.removeprefix("fake://")])
        return True
    monkeypatch.setattr(dl, "fetch", fetch)


def test_stop_after_tries_smallest_first_and_stops_at_first_success(fake, monkeypatch, capsys):
    _two_archive_world(monkeypatch, fake, {
        "Big_1of1.zip": ({"Animals/1. Dog/MVI_3030.MOV": b"d"}, 5000),
        "Small_1of1.zip": ({"Other/9. x/none.MOV": b"n"}, 0),          # smallest, but nothing wanted
        "Mid_1of1.zip": ({"Electronics/53. Fan/MVI_4532.MOV": b"f"}, 1000),
        "Huge_1of1.zip": ({"Animals/1. Dog/MVI_3030.MOV": b"d"}, 90000)})
    code = run(fake, "--all", "--delete-zips", "--confirm-reviewed-rights", "--stop-after", "1", "--max-archives", "5")
    assert code == 0 and fake["fetches"] == ["Small_1of1.zip", "Mid_1of1.zip"]
    assert "STOPPED EARLY" in capsys.readouterr().out


def test_max_archives_bounds_a_smoke_test_that_never_matches(fake, monkeypatch, capsys):
    _two_archive_world(monkeypatch, fake, {f"A{i}_1of1.zip": ({"Other/9. x/none.MOV": b"n"}, i * 10) for i in range(6)})
    code = run(fake, "--all", "--delete-zips", "--confirm-reviewed-rights", "--stop-after", "1", "--max-archives", "3")
    assert code == 2 and len(fake["fetches"]) == 3
    assert "GAVE UP" in capsys.readouterr().out


# ---- --inspect all: one cheap pre-flight across every archive ----------------------------------------------

class _FakeRange(io.BytesIO):
    """Stands in for RangeFile: serves an in-memory archive and counts nothing."""
    registry: dict[str, bytes] = {}

    def __init__(self, url, size=None, **kwargs):
        super().__init__(self.registry[url.removeprefix("fake://")])
        self.bytes_fetched, self.requests = 0, 1


def test_inspect_all_sums_matches_across_archives(fake, monkeypatch, capsys):
    _FakeRange.registry = dict(ARCHIVES)
    monkeypatch.setattr(dl, "RangeFile", _FakeRange)
    assert dl.main(["--inspect", "all", "--keep", fake["keep"]]) == 0
    out = capsys.readouterr().out
    assert "INSPECTED 2 archives" in out and "exact=2" in out and "wanted=2" in out


def test_inspect_reports_unavailable_when_server_ignores_ranges(fake, monkeypatch, capsys):
    def refuse(url, size=None, **kwargs):
        raise dl.RangeNotSupported("server answered HTTP 200 to a Range request")
    monkeypatch.setattr(dl, "RangeFile", refuse)
    assert dl.main(["--inspect", "all", "--keep", fake["keep"]]) == 3
    assert "INSPECT UNAVAILABLE" in capsys.readouterr().out
