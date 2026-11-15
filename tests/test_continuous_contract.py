"""Dependency-light tests for continuous task metadata and text contracts."""
import json
import unittest

from ml.continuous.text import BOS, EOS, UNK, build_vocabulary, decode, encode, normalize_for_model
from ml.datasets.adapters import adapt_continuous_records, adapt_include_records


class ContinuousContractTests(unittest.TestCase):
    def test_vocabulary_is_built_from_supplied_training_references(self):
        vocabulary = build_vocabulary(["Hello world", "hello ISL"], max_size=16)
        self.assertEqual(vocabulary["hello"], 4)
        self.assertIn(UNK, vocabulary)
        encoded = encode("hello missing", vocabulary)
        self.assertEqual(encoded[0], vocabulary[BOS])
        self.assertEqual(encoded[-1], vocabulary[EOS])
        self.assertEqual(decode(encoded, vocabulary), "hello")
        self.assertEqual(normalize_for_model("  HELLO\tWorld "), "hello world")

    def test_continuous_adapter_keeps_parent_video_and_original_references(self):
        records = [
            {"clip": "clips/a.mp4", "sentence": "  The signer   greets. ", "vid": "video-a", "ann": "a-1", "split": "train"},
            {"clip": "clips/a.mp4", "sentence": "Signer greets", "vid": "video-a", "ann": "a-2", "split": "train"},
        ]
        rows = adapt_continuous_records(
            records, dataset_id="corpus", dataset_version="v2", media_field="clip",
            reference_field="sentence", video_id_field="vid", sample_id_field="ann",
            split_field="split", license_id="terms:review-1", rights_status="approved_research",
        )
        self.assertNotEqual(rows[0]["sample_id"], rows[1]["sample_id"])
        self.assertEqual(rows[0]["group_id"], rows[1]["group_id"])
        self.assertEqual(json.loads(rows[0]["reference_texts_json"]), ["The signer   greets."])
        self.assertEqual(json.loads(rows[0]["normalized_references_json"]), ["the signer greets."])

    def test_include_adapter_does_not_assume_rights_or_parse_string_false_as_true(self):
        rows = adapt_include_records(
            [{"video_path": "colors/yellow/v1.mp4", "label": "yellow", "split": "validation",
              "include_50": "false"}], dataset_version="release-1", include_50_only=False,
        )
        self.assertEqual(rows[0]["task"], "isolated")
        self.assertEqual(rows[0]["split"], "val")
        self.assertEqual(rows[0]["rights_status"], "unknown")
        self.assertEqual(json.loads(rows[0]["metadata_json"])["include_50"], False)


if __name__ == "__main__":
    unittest.main()
