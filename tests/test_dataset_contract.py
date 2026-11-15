"""Standard-library tests for canonical data provenance and split guards."""
import json
import os
import tempfile
import unittest
from pathlib import Path

from ml.datasets.canonical import ManifestError, make_row, resolve_media_path, safe_relative_uri, stable_sample_id
from ml.datasets.validate import validate_rows
from ml.datasets.near_duplicates import _signature_distance
from ml.datasets.fingerprint import fingerprint_manifest


class CanonicalManifestTests(unittest.TestCase):
    def tempdir(self):
        return tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])

    def row(self, *, task="isolated", sample="sample-1", split="train", group="signer-1", **extra):
        return make_row(
            dataset_id="synthetic-fixture", dataset_version="v1", source_sample_id=sample,
            task=task, media_uri=f"videos/{sample}.mp4", split=split, group_id=group,
            rights_status="approved_research", label_id="HELLO" if task == "isolated" else "",
            source_video_id=sample if task == "continuous" else "",
            reference_texts=["ನಮಸ್ಕಾರ"] if task == "continuous" else [], **extra,
        )

    def test_stable_identity_and_safe_paths(self):
        a = stable_sample_id("d", "v1", "x")
        self.assertEqual(a, stable_sample_id("d", "v1", "x"))
        self.assertNotEqual(a, stable_sample_id("d", "v2", "x"))
        for bad in ("../secret.mp4", "C:/secret.mp4", "/root/secret.mp4", "a/../../b.mp4", "a\x00b"):
            with self.subTest(path=bad), self.assertRaises(ManifestError):
                safe_relative_uri(bad)

    def test_schema_rejects_identity_tampering_and_bad_references(self):
        row = self.row(task="continuous")
        row["sample_id"] = "forged"
        row["reference_texts_json"] = "{}"
        codes = {item["code"] for item in __import__("ml.datasets.canonical", fromlist=["validate_row_shape"]).validate_row_shape(row)}
        self.assertIn("SAMPLE_ID_MISMATCH", codes)
        self.assertIn("REFERENCE_JSON_INVALID", codes)

    def test_group_leakage_and_missing_media_are_errors(self):
        rows = [self.row(sample="one", split="train"), self.row(sample="two", split="test")]
        with self.tempdir() as temp:
            result = validate_rows(rows, data_root=temp, require_approved_rights=True,
                                   require_splits=True, require_group_ids=True)
        codes = {item["code"] for item in result["issues"]}
        self.assertIn("GROUP_SPLIT_LEAKAGE", codes)
        self.assertIn("MEDIA_NOT_FOUND", codes)

    def test_media_symlink_cannot_escape_data_root(self):
        with self.tempdir() as temp, self.tempdir() as other:
            root = Path(temp) / "root"
            root.mkdir()
            target = Path(other) / "external.mp4"
            target.write_bytes(b"fixture")
            link = root / "escape.mp4"
            try:
                link.symlink_to(target)
            except (OSError, NotImplementedError):
                self.skipTest("symlinks unavailable")
            with self.assertRaises(ManifestError):
                resolve_media_path(str(root), "escape.mp4")

    def test_continuous_requires_parent_and_reference(self):
        row = self.row(task="continuous")
        row["source_video_id"] = ""
        row["reference_texts_json"] = json.dumps([])
        from ml.datasets.canonical import validate_row_shape
        codes = {item["code"] for item in validate_row_shape(row)}
        self.assertIn("SOURCE_VIDEO_ID_MISSING", codes)
        self.assertIn("REFERENCE_TEXT_MISSING", codes)

    def test_visual_hash_distance_is_candidate_metric_only(self):
        identical = (0x1234, 0xAABB, 0x55AA)
        self.assertEqual(_signature_distance(identical, identical), 0.0)
        one_bit = (0x1235, 0xAABB, 0x55AA)
        self.assertGreater(_signature_distance(identical, one_bit), 0.0)
        self.assertLess(_signature_distance(identical, one_bit), 0.01)

    def test_fingerprint_records_sha256_without_mutating_source_row(self):
        with self.tempdir() as temp:
            root = Path(temp) / "data"
            (root / "videos").mkdir(parents=True)
            media = root / "videos" / "sample-1.mp4"
            media.write_bytes(b"synthetic metadata fixture, not a video")
            source = self.row()
            result = fingerprint_manifest([source], str(root))[0]
        self.assertEqual(source["sha256"], "")
        self.assertEqual(len(result["sha256"]), 64)
        self.assertEqual(int(result["size_bytes"]), len(b"synthetic metadata fixture, not a video"))


if __name__ == "__main__":
    unittest.main()
