import json, os, time
import numpy as np
import pytest
from fastapi.testclient import TestClient
from ml.audit.inspect_project import dataset_inventory, scan_tree
from ml.datasets import manifest as M
from ml.evaluation import metrics as K
from ml.evaluation.word_eval import evaluate_rows
from ml.inference.subtitles import to_srt, to_txt, to_vtt
from ml.test_library.build import build_library


def make_tree(tmp, per=8):
    """tmp/<cat>/<N. word>/MVI_<w>_<n>.MOV ; numbers form 4 sessions per category (gaps > 30)."""
    for ci, cat in enumerate(("Animals", "Colours")):
        for w in ("1. Dog", "2. Cat", "3. Cow"):
            d = tmp / cat / w; d.mkdir(parents=True)
            for i in range(per):
                n = 1000 * ci + 100 + (i % 2) + (i // 2) * 200
                (d / f"MVI_{w[0]}_{n}.MOV").write_bytes(os.urandom(2048) + bytes([i, ci]))
    return tmp


def rows_of(tmp):
    return M.add_fingerprints(M.infer_sessions(M.from_isolated_tree(str(tmp))))


def test_manifest_sessions_split_no_leakage(tmp_path):
    rows = rows_of(make_tree(tmp_path))
    assert len(rows) == 48 and {r["label"] for r in rows} == {"dog", "cat", "cow"}
    assert len({r["session_id"] for r in rows}) == 8
    M.make_group_split(rows, seed=1)
    assert M.check_leakage(rows).get("groups_in_multiple_splits") is None
    assert M.summary(rows)["videos"] == 48


def test_leakage_checker_detects_problems(tmp_path):
    rows = rows_of(make_tree(tmp_path)); M.make_group_split(rows)
    rows[0]["split"] = "train"; rows[1]["split"] = "test"; rows[1]["session_id"] = rows[0]["session_id"]
    rows[2]["sha1_head"], rows[2]["n_bytes"] = rows[3]["sha1_head"], rows[3]["n_bytes"]; rows[2]["split"], rows[3]["split"] = "train", "test"
    bad = M.check_leakage(rows)
    assert "groups_in_multiple_splits" in bad and "identical_files_across_splits" in bad


def test_metrics_known_values():
    assert K.wer("a b c", "a x c") == pytest.approx(1 / 3) and K.wer("a b", "a b") == 0
    assert K.corpus_wer(["a b", "c d"], ["a b", "c x"]) == pytest.approx(0.25)
    assert K.bleu(["the cat sat on the mat"], ["the cat sat on the mat"]) == pytest.approx(100)
    assert K.chrf(["hello world"], ["hello world"]) == pytest.approx(100)
    assert K.rouge_l("a b c d", "a c d") == pytest.approx(2 * 1 * .75 / 1.75)
    cm = K.confusion_matrix([0, 0, 1, 1], [0, 1, 1, 1], 2); rep, macro = K.per_class_report(cm, ["a", "b"])
    assert rep["a"]["recall"] == 0.5 and rep["b"]["precision"] == pytest.approx(2 / 3) and 0 < macro["f1"] < 1
    assert K.ece([1.0, 1.0], [1, 0]) == pytest.approx(0.5)
    sc = K.selective_curve([0.9, 0.2], [1, 0], (0.0, 0.5)); assert sc[1]["coverage"] == 0.5 and sc[1]["accuracy"] == 1.0
    with pytest.raises(ValueError): K.wer("", "x")


def test_subtitles():
    segs = [{"start": 0, "end": 1.5, "text": "hello"}, {"start": 2, "end": 3.25, "text": "world"}]
    assert to_srt(segs).startswith("1\n00:00:00,000 --> 00:00:01,500\nhello") and "00:00:03,250" in to_srt(segs)
    assert to_vtt(segs).startswith("WEBVTT") and "00:00:02.000 --> 00:00:03.250" in to_vtt(segs)
    assert to_txt(segs) == "hello\nworld\n"
    with pytest.raises(ValueError): to_srt([{"start": 2, "end": 1, "text": "x"}])


def test_test_library_only_heldout_and_no_duplicates(tmp_path):
    src = make_tree(tmp_path / "src"); rows = rows_of(src); M.make_group_split(rows, seed=2)
    out = build_library(rows, str(tmp_path / "lib"), per_class=2)
    assert out and all(r["split"] == "test" for r in out)
    test_sources = {r["sample_id"] for r in rows if r["split"] == "test"}
    assert {r["source_sample_id"] for r in out} <= test_sources
    assert os.path.exists(tmp_path / "lib" / "manifests" / "test_manifest.csv")
    assert os.path.exists(tmp_path / "lib" / "sentence_translation" / "README.txt")
    assert len({os.path.getsize(tmp_path / "lib" / r["file"]) for r in out}) >= 1
    # forbidden: a held-out file byte-identical to a training file
    rows2 = [dict(r) for r in rows]; t = next(r for r in rows2 if r["split"] == "test"); tr = next(r for r in rows2 if r["split"] == "train")
    tr["sha1_head"], tr["n_bytes"] = t["sha1_head"], t["n_bytes"]
    with pytest.raises(ValueError): build_library(rows2, str(tmp_path / "lib2"), per_class=50)


def test_word_eval_outputs_and_errors(tmp_path):
    labels = ["dog", "cat", "idle"]
    rows = [dict(sample_id=f"s{i}", dataset="d", label=l, path=f"/x/{i}") for i, l in enumerate(["dog", "dog", "cat", "cat", "bird"])]
    def predict(path):
        i = int(path.split("/")[-1])
        if i == 3: raise RuntimeError("corrupt video")
        pred = ["dog", "cat", "cat", "cat", "dog"][i]
        return {"label": pred, "conf": 0.9 if i != 1 else 0.3, "top": [[pred, .9], ["dog", .05], ["cat", .03], ["idle", .01], ["x", 0]], "status": "confident", "ms": 5.0}
    s, recs = evaluate_rows(rows, predict, labels, str(tmp_path / "rep"))
    assert s["n_scored"] == 3 and s["n_errors"] == 1 and s["n_out_of_vocabulary"] == 1
    assert s["top1"] == pytest.approx(2 / 3) and "NOT evidence" in s["note"]
    for f in ("predictions.csv", "classification_report.json", "confusion_matrix.png", "error_analysis.html", "eval_summary.json"):
        assert os.path.getsize(tmp_path / "rep" / f) > 0
    assert "corrupt video" in open(tmp_path / "rep" / "error_analysis.html").read()


def test_audit_scan(tmp_path):
    make_tree(tmp_path / "data"); (tmp_path / "a.ipynb").write_text("{}"); (tmp_path / "test_x.py").write_text("")
    (tmp_path / "m.onnx").write_bytes(b"1"); (tmp_path / "requirements.txt").write_text("")
    t = scan_tree(str(tmp_path)); assert t["videos"] == 48 and len(t["tests"]) == 1 and len(t["kinds"]["checkpoint"]) == 1
    inv = dataset_inventory(str(tmp_path / "data")); assert inv["videos"] == 48 and inv["folders_with_videos"] == 6 and inv["median"] == 8


# ---------------- API ----------------
class StubPipe:
    class predictor: labels = ["dog", "cat", "idle"]
    def __init__(self, status="confident"): self.status = status
    def predict_video(self, path):
        if os.path.getsize(path) == 4096: raise ValueError("cannot decode")
        return {"label": "dog", "conf": .93, "margin": .8, "top": [["dog", .93], ["cat", .05]], "status": self.status,
                "quality": {}, "duration_s": 2.0, "ms": 12.0}


@pytest.fixture
def api(tmp_path, monkeypatch):
    import backend.api_ml as A, backend.main as m, backend.media as Media
    rows = rows_of(make_tree(tmp_path / "src")); M.make_group_split(rows, seed=3)
    lib = build_library(rows, str(tmp_path / "test_videos"), per_class=1)
    for k, v in dict(TEST_ROOT=tmp_path / "test_videos", UPLOADS=tmp_path / "up", DB=tmp_path / "jobs.db", REPORTS=tmp_path / "rep").items():
        monkeypatch.setattr(A, k, v)
    monkeypatch.setattr(Media, "probe_video", lambda path, limits: {
        "size_bytes": os.path.getsize(path), "width": 640, "height": 480, "fps": 30.0,
        "declared_frames": 60, "decoded_frames": 60, "duration_s": 2.0,
        "fourcc": "mp4v", "frame_count_metadata_mismatch": False,
    })
    m.app.state.pipeline = StubPipe()
    yield TestClient(m.app), lib, tmp_path
    m.app.state.pipeline = None


def test_api_library_and_predict(api):
    c, lib, _ = api
    items = c.get("/api/test-library").json()["items"]; assert len(items) == len(lib) > 0
    sid = items[0]["id"]
    assert c.get(f"/api/test-library/{sid}/video").status_code == 200
    r = c.post(f"/api/test-library/{sid}/predict").json()
    assert r["ground_truth"] == items[0]["label"] and r["correct"] == (r["label"] == items[0]["label"]) and r["ms"] == 12.0
    assert c.get("/api/test-library/nope/video").status_code == 404


def test_api_upload_validation_and_jobs(api):
    c, _, _ = api
    assert c.post("/api/predict", files={"file": ("x.txt", b"a" * 5000)}).status_code == 415
    assert c.post("/api/predict", files={"file": ("x.mp4", b"a")}).status_code == 400
    assert c.post("/api/predict", files={"file": ("x.mp4", b"a" * 4096)}).status_code == 422
    ok = c.post("/api/predict", files={"file": ("x.mp4", os.urandom(5000))}); assert ok.status_code == 200 and ok.json()["label"] == "dog"
    data = os.urandom(6000)
    j = c.post("/api/jobs", files={"file": ("clip.mp4", data)}).json(); assert j["status"] == "queued" and not j["reused"]
    for _ in range(50):
        st = c.get(f"/api/jobs/{j['id']}").json()
        if st["status"] in ("done", "failed"): break
        time.sleep(0.05)
    assert st["status"] == "done" and st["result"]["label"] == "dog"
    again = c.post("/api/jobs", files={"file": ("clip.mp4", data)}).json(); assert again["reused"] and again["id"] == j["id"]
    assert "dog" in c.get(f"/api/jobs/{j['id']}/export?format=txt").text
    assert c.get(f"/api/jobs/{j['id']}/export?format=srt").text.startswith("1\n00:00:00,000")
    assert c.get(f"/api/jobs/{j['id']}/export?format=vtt").text.startswith("WEBVTT")
    assert c.get(f"/api/jobs/{j['id']}/export?format=doc").status_code == 400
    assert c.get("/api/jobs/missing").status_code == 404


def test_api_unsure_not_exported_and_no_model_503_and_translate_501(api):
    import backend.main as m
    c, _, _ = api
    m.app.state.pipeline = StubPipe("uncertain")
    j = c.post("/api/jobs", files={"file": ("u.mp4", os.urandom(7000))}).json()
    for _ in range(50):
        if c.get(f"/api/jobs/{j['id']}").json()["status"] == "done": break
        time.sleep(0.05)
    assert c.get(f"/api/jobs/{j['id']}/export?format=txt").status_code == 409
    m.app.state.pipeline = None
    assert c.post("/api/predict", files={"file": ("x.mp4", os.urandom(5000))}).status_code == 503
    assert c.get("/api/model").json()["word_model_loaded"] is False
    assert c.post("/api/translate").status_code == 503


def test_api_reports(api):
    c, _, tmp = api
    assert c.get("/api/reports/summary").status_code == 404
    (tmp / "rep").mkdir(); (tmp / "rep" / "eval_summary.json").write_text(json.dumps({"top1": .5}))
    assert c.get("/api/reports/summary").json()["top1"] == .5
    assert c.get("/api/reports/file/../secret").status_code in (404, 422) and c.get("/api/reports/file/passwd").status_code == 404


def test_filter_keep_and_key3(tmp_path):
    rows = M.from_isolated_tree(str(make_tree(tmp_path)))
    one = M.key3(rows[0]["path"]); assert one == one.lower() and one.count("/") == 2
    assert [r["path"] for r in M.filter_keep(rows, {one})] == [rows[0]["path"]]
    assert M.key3("A\\B 1\\C.MOV") == "a/b 1/c.mov"
