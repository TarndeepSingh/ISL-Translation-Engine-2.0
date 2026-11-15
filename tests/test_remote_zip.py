"""Remote ZIP reading over HTTP Range, tested against a REAL local HTTP server (no mocks of urllib).

The server counts requests and bytes, can ignore Range (like a misconfigured host), and can fail the first N
requests. Archives hold random placeholder bytes, so incompressible data makes byte counts meaningful.
"""
from __future__ import annotations

import io
import os
import re
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import isl.download_include as dl
from isl.remote_zip import RangeFile, RangeNotSupported

KEEP = ["Electronics/53. Fan/MVI_4532.MOV", "Animals/1. Dog/MVI_3030.MOV"]


def make_archive(extra_members=0):
    members = {"top/electronics/53. fan/mvi_4532.mov": os.urandom(300_000),      # wanted (prefix + other case)
               "top/Animals/1. Dog/MVI_3030.MOV": os.urandom(250_000)}            # wanted
    for i in range(extra_members):
        members[f"top/Animals/9. Cat/MVI_{i}.MOV"] = os.urandom(400_000)          # not wanted
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as bundle:
        for name, data in members.items():
            bundle.writestr(name, data)
    return buffer.getvalue(), members


class State:
    data = b""
    honour_range = True
    fail_next = 0
    requests = 0
    bytes_sent = 0


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass

    def do_GET(self):
        State.requests += 1
        if State.fail_next > 0:
            State.fail_next -= 1
            self.send_response(503); self.end_headers(); return
        data, header = State.data, self.headers.get("Range")
        match = re.match(r"bytes=(\d+)-(\d*)", header or "")
        if not (State.honour_range and match):
            self.send_response(200); self.send_header("Content-Length", str(len(data))); self.end_headers()
            self.wfile.write(data); State.bytes_sent += len(data); return
        start = int(match.group(1)); end = min(int(match.group(2) or len(data) - 1), len(data) - 1)
        body = data[start:end + 1]
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
        self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body); State.bytes_sent += len(body)


@pytest.fixture
def server():
    State.honour_range, State.fail_next, State.requests, State.bytes_sent = True, 0, 0, 0
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/a.zip"
    httpd.shutdown(); httpd.server_close()


def test_lists_members_without_downloading_the_archive(server):
    State.data, members = make_archive(extra_members=20)       # ~8 MB archive
    remote = RangeFile(server, len(State.data), sleep=lambda s: None)
    with zipfile.ZipFile(remote) as archive:
        assert sorted(archive.namelist()) == sorted(members)
    assert remote.bytes_fetched < 0.2 * len(State.data)         # only the index, not the 8 MB of videos


def test_size_is_probed_from_content_range_when_not_given(server):
    State.data, _ = make_archive()
    remote = RangeFile(server, sleep=lambda s: None)
    assert remote.size == len(State.data)


def test_server_that_ignores_range_raises_and_never_pulls_the_body(server):
    State.data, _ = make_archive(extra_members=10)
    State.honour_range = False
    with pytest.raises(RangeNotSupported):
        RangeFile(server, sleep=lambda s: None)
    with pytest.raises(RangeNotSupported):
        remote = RangeFile(server, len(State.data), sleep=lambda s: None)
        remote.seek(0); remote.read(10)


def test_transient_server_errors_are_retried(server):
    State.data, members = make_archive()
    State.fail_next = 2
    remote = RangeFile(server, len(State.data), tries=5, sleep=lambda s: None)
    with zipfile.ZipFile(remote) as archive:
        assert sorted(archive.namelist()) == sorted(members)


def test_persistent_failure_gives_a_clear_error(server):
    State.data, _ = make_archive()
    State.fail_next = 100
    remote = RangeFile(server, len(State.data), tries=3, sleep=lambda s: None)
    with pytest.raises(OSError, match="failed after 3 tries"):
        remote.seek(0); remote.read(10)


def test_extracts_only_wanted_members_at_manifest_paths_fetching_little(server, tmp_path):
    State.data, members = make_archive(extra_members=30)       # ~12 MB of unwanted videos
    remote = RangeFile(server, len(State.data), sleep=lambda s: None)
    out = tmp_path / "raw"; out.mkdir()
    assert dl.safe_extract_zip(remote, out, KEEP) == 2
    assert (out / "Electronics/53. Fan/MVI_4532.MOV").read_bytes() == members["top/electronics/53. fan/mvi_4532.mov"]
    assert (out / "Animals/1. Dog/MVI_3030.MOV").read_bytes() == members["top/Animals/1. Dog/MVI_3030.MOV"]
    wanted_bytes = 300_000 + 250_000
    assert remote.bytes_fetched < wanted_bytes * 1.6 + 200_000   # not the ~12.5 MB archive
    assert remote.bytes_fetched < 0.25 * len(State.data)


def test_corrupted_range_is_caught_by_member_crc(server, tmp_path):
    data, members = make_archive()
    start = data.index(members["top/Animals/1. Dog/MVI_3030.MOV"][:64])
    State.data = data[:start + 100] + bytes([data[start + 100] ^ 0xFF]) + data[start + 101:]
    remote = RangeFile(server, len(State.data), sleep=lambda s: None)
    out = tmp_path / "raw"; out.mkdir()
    with pytest.raises(zipfile.BadZipFile):
        dl.safe_extract_zip(remote, out, KEEP)
    assert not (out / "Animals/1. Dog/MVI_3030.MOV").exists()    # partial file removed


def test_inspect_archive_reports_layout_mismatch_that_exact_matching_misses():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.writestr("flat/MVI_4532.MOV", b"x")                  # layout without category/label folders
        bundle.writestr("Animals/1. Dog/MVI_3030.MOV", b"y")
    info = dl.inspect_archive(buffer, KEEP)
    assert info["members"] == 2 and info["exact"] == 1 and info["name_only"] == 2


def test_main_remote_mode_extracts_and_marks_done_then_skips(server, tmp_path, monkeypatch, capsys):
    State.data, _ = make_archive(extra_members=10)
    monkeypatch.setattr(dl, "list_archives", lambda api=None: [("A_1of1.zip", server, len(State.data), "")])
    keep = tmp_path / "keep.txt"; keep.write_text("\n".join(KEEP) + "\n")
    out = tmp_path / "raw"
    args = ["--all", "--remote", "--keep", str(keep), "--out", str(out), "--confirm-reviewed-rights"]
    assert dl.main(args) == 0
    assert "extracted 2 files (fetched" in capsys.readouterr().out
    assert not list(out.glob("*.zip"))
    before = State.requests
    assert dl.main(args) == 0
    assert State.requests == before                               # finished archive: no network at all
    assert "already done (no archive kept)" in capsys.readouterr().out


def test_main_remote_falls_back_to_full_download_when_range_is_ignored(server, tmp_path, monkeypatch, capsys):
    State.data, _ = make_archive()
    State.honour_range = False
    monkeypatch.setattr(dl, "list_archives", lambda api=None: [("A_1of1.zip", server, len(State.data), "")])
    fetched = []

    def fake_fetch(url, dst, size, checksum="", tries=12):
        fetched.append(url)
        open(dst, "wb").write(State.data)
        return True
    monkeypatch.setattr(dl, "fetch", fake_fetch)
    keep = tmp_path / "keep.txt"; keep.write_text("\n".join(KEEP) + "\n")
    out = tmp_path / "raw"
    assert dl.main(["--all", "--remote", "--delete-zips", "--keep", str(keep), "--out", str(out),
                    "--confirm-reviewed-rights"]) == 0
    text = capsys.readouterr().out
    assert "falling back to a full download" in text and fetched
    assert (out / "Animals/1. Dog/MVI_3030.MOV").exists()


def test_main_inspect_prints_layout_and_counts_without_extracting(server, tmp_path, monkeypatch, capsys):
    State.data, _ = make_archive(extra_members=5)
    monkeypatch.setattr(dl, "list_archives", lambda api=None: [("A_1of1.zip", server, len(State.data), "")])
    keep = tmp_path / "keep.txt"; keep.write_text("\n".join(KEEP) + "\n")
    assert dl.main(["--inspect", "A_1of1", "--keep", str(keep), "--out", str(tmp_path / "raw")]) == 0
    text = capsys.readouterr().out
    assert "matches by last-3 parts=2" in text and "top/" in text
    assert not (tmp_path / "raw").exists()
