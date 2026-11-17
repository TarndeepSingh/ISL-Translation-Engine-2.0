import json

import numpy as np
import pytest

from ml.evaluation import class_report as cr

LABELS = ["1. Dog", "2. Death", "3. quiet"]


def make_run(root, name, y, p, labels=LABELS, top1=None):
    d = root / name
    d.mkdir(parents=True)
    np.save(d / "test_pred.npy", np.stack([np.array(y), np.array(p)]))
    (d / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
    if top1 is not None:
        (d / "experiment.json").write_text(json.dumps({"test_top1": top1, "test_top3": 1.0}), encoding="utf-8")
    return d


Y = [0, 0, 0, 0, 1, 1, 2, 2, 2]
P = [0, 0, 0, 1, 1, 1, 2, 0, 2]   # class0 3/4, class1 2/2, class2 2/3 ; one 0->1 and one 2->0 error


def test_known_answer(tmp_path):
    make_run(tmp_path, "include50-balanced-aug7-warp0.3-seed13", Y, P, top1=7 / 9)
    out = tmp_path / "isl.eval.json"
    rep = cr.main(["--runs-dir", str(tmp_path), "--seeds", "13", "--out", str(out), "--plots", ""])
    d = rep["deployed"]
    assert d["n_test"] == 9 and d["correct"] == 7
    assert d["overall_accuracy"] == pytest.approx(7 / 9)
    assert d["average_class_accuracy"] == pytest.approx((0.75 + 1.0 + 2 / 3) / 3)   # differs from overall on purpose
    acc = {c["word"]: c["accuracy"] for c in rep["per_class"]}
    assert acc == {"Dog": 0.75, "Death": 1.0, "quiet": pytest.approx(2 / 3)}
    assert rep["confusion_matrix"] == [[3, 1, 0], [0, 2, 0], [1, 0, 2]]
    assert {(c["true"], c["pred"], c["count"]) for c in rep["top_confusions"]} == {("Dog", "Death", 1), ("quiet", "Dog", 1)}
    assert rep["warnings"] == []
    assert json.loads(out.read_text(encoding="utf-8"))["schema_version"] == cr.SCHEMA


def test_warns_when_experiment_json_disagrees(tmp_path):
    make_run(tmp_path, "include50-balanced-aug7-warp0.3-seed13", Y, P, top1=0.99)
    rep = cr.main(["--runs-dir", str(tmp_path), "--seeds", "13", "--out", str(tmp_path / "o.json"), "--plots", ""])
    assert any("differs from experiment.json" in w for w in rep["warnings"])


def test_seed_summary_and_missing_seed_skipped(tmp_path):
    make_run(tmp_path, "include50-balanced-aug7-warp0.3-seed13", Y, P)
    make_run(tmp_path, "include50-balanced-aug7-warp0.3-seed14", Y, Y)      # perfect second seed
    rep = cr.main(["--runs-dir", str(tmp_path), "--seeds", "13,14,15", "--out", str(tmp_path / "o.json"), "--plots", ""])
    assert [s["seed"] for s in rep["seeds"]] == [13, 14]                     # seed 15 absent -> skipped, not fatal
    assert rep["seed_summary"]["overall_mean"] == pytest.approx((7 / 9 + 1.0) / 2)
    dog = rep["per_class"][0]
    assert dog["seed_accuracy"] == {"13": 0.75, "14": 1.0}


def test_different_label_lists_are_refused(tmp_path):
    make_run(tmp_path, "include50-balanced-aug7-warp0.3-seed13", Y, P)
    make_run(tmp_path, "include50-balanced-aug7-warp0.3-seed14", Y, Y, labels=["a", "b", "c"])
    with pytest.raises(ValueError, match="different class list"):
        cr.main(["--runs-dir", str(tmp_path), "--seeds", "13,14", "--out", str(tmp_path / "o.json"), "--plots", ""])


def test_index_outside_label_range_is_refused(tmp_path):
    d = make_run(tmp_path, "run", [0, 1], [0, 5])
    with pytest.raises(ValueError, match="do not belong together"):
        cr.load_run(d)


def test_clean_word_and_wilson():
    assert cr.clean_word("61. Father") == "Father"
    assert cr.clean_word("1. loud") == "loud"
    assert cr.clean_word("Greetings: Hello") == "Greetings: Hello"      # not a numeric prefix -> untouched
    assert cr.wilson(0, 0) == (0.0, 0.0)
    lo, hi = cr.wilson(2, 2)
    assert hi == pytest.approx(1.0) and 0.3 < lo < 0.4                 # 2/2 is NOT proof of 100%
    lo, hi = cr.wilson(190, 200)
    assert 0.91 < lo < 0.93 and 0.96 < hi < 0.98


def test_plots_are_written(tmp_path):
    make_run(tmp_path, "include50-balanced-aug7-warp0.3-seed13", Y, P)
    cr.main(["--runs-dir", str(tmp_path), "--seeds", "13", "--out", str(tmp_path / "o.json"), "--plots", str(tmp_path / "plots")])
    names = sorted(p.name for p in (tmp_path / "plots").iterdir())
    assert names == ["confusion_matrix_counts.png", "confusion_matrix_normalized.png", "per_class_accuracy.png"]
