import csv
import json
from pathlib import Path

import numpy as np
import pytest

MODEL = Path("models/isl.onnx")
LABELS = Path("models/isl.labels.json")
pytestmark = pytest.mark.skipif(not (MODEL.exists() and LABELS.exists()), reason="needs the exported model in models/")


def test_every_video_is_scored_exactly_once(tmp_path):
    from ml.evaluation import split_report as sr

    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    rng = np.random.default_rng(0)
    rows = []
    for i in range(30):
        f = tmp_path / f"s{i}.npz"
        np.savez(f, raw=rng.normal(0.5, 0.1, (40, 75, 3)).astype(np.float32))
        rows.append({"sample_id": f"s{i}", "cache": str(f), "label": labels[i % 50], "split": ["train", "val", "test"][i % 3]})
    with open(tmp_path / "index.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    rep = sr.main(["--index", str(tmp_path / "index.csv"), "--onnx", str(MODEL), "--labels", str(LABELS), "--out", str(tmp_path / "o.json")])
    assert sum(rep["splits"][s]["n"] for s in sr.SPLITS) == 30
    for s in sr.SPLITS:
        d = rep["splits"][s]
        assert sum(c["support"] for c in d["per_class"]) == d["n"]
        assert all(0 <= c["correct"] <= c["support"] for c in d["per_class"])


def test_wrong_label_file_is_refused(tmp_path):
    from ml.evaluation import split_report as sr

    (tmp_path / "labels.json").write_text(json.dumps(["a", "b"]), encoding="utf-8")
    np.savez(tmp_path / "x.npz", raw=np.zeros((10, 75, 3), np.float32))
    with open(tmp_path / "index.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["sample_id", "cache", "label", "split"])
        w.writeheader()
        w.writerow({"sample_id": "1", "cache": str(tmp_path / "x.npz"), "label": "1. Dog", "split": "train"})
    with pytest.raises(SystemExit, match="not in"):
        sr.main(["--index", str(tmp_path / "index.csv"), "--onnx", str(MODEL), "--labels", str(tmp_path / "labels.json"), "--out", str(tmp_path / "o.json")])
