"""Train an experimental continuous pose-to-English Transformer baseline.

Requires a rights-approved continuous `index.csv` from ml.datasets.preprocess.
Example: python -m ml.continuous.train --index data/processed/continuous-v1/index.csv --out runs/continuous/exp1
This baseline uses the existing 170D pose/hand features and does not claim to
represent facial/non-manual cues. Never use it as evidence of production quality.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from isl.config import FEATURE_DIM
from isl.features import normalize, resample
from .evaluation import corpus_metrics
from .model import VideoToTextTransformer
from .text import BOS, EOS, PAD, build_vocabulary, decode, encode


def load_rows(index_path: str) -> list[dict[str, str]]:
    with open(index_path, newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    required = {"cache", "sample_id", "task", "group_id", "source_video_id", "split",
                "reference_texts_json", "sha256", "license_id", "rights_status"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"index must have rows and columns: {', '.join(sorted(required))}")
    seen = set()
    base = os.path.dirname(os.path.abspath(index_path))
    for row in rows:
        if row["sample_id"] in seen or not row["sample_id"]:
            raise ValueError("sample_id values must be present and unique")
        seen.add(row["sample_id"])
        if row["task"] != "continuous":
            raise ValueError("continuous training index contains a different task")
        if row["split"] not in {"train", "val", "test"}:
            raise ValueError(f"sample {row['sample_id']} has no train/val/test split")
        if not row["source_video_id"] or not row["group_id"]:
            raise ValueError(f"sample {row['sample_id']} lacks source-video/group identity")
        if row["rights_status"] not in {"approved_research", "approved_noncommercial", "approved_commercial"}:
            raise ValueError(f"sample {row['sample_id']} lacks an approved rights status")
        if not row["license_id"].strip():
            raise ValueError(f"sample {row['sample_id']} has no license/source-terms reference")
        digest = row["sha256"].lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError(f"sample {row['sample_id']} has no valid source SHA-256")
        if row.get("preprocess_version") != "mediapipe-tasks-pose-hands-75x3-v1":
            raise ValueError(f"sample {row['sample_id']} has unsupported landmark preprocessing")
        try:
            refs = json.loads(row["reference_texts_json"])
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid references for {row['sample_id']}") from exc
        if not isinstance(refs, list) or not any(str(value).strip() for value in refs):
            raise ValueError(f"sample {row['sample_id']} has no usable English reference")
        if not os.path.isabs(row["cache"]):
            row["cache"] = os.path.abspath(os.path.join(base, row["cache"]))
        if not os.path.isfile(row["cache"]):
            raise ValueError(f"cache artifact is missing for {row['sample_id']}")
    return rows


def validate_splits(rows: list[dict[str, str]]) -> None:
    groups: dict[str, set[str]] = {}
    videos: dict[str, set[str]] = {}
    hashes: dict[str, set[str]] = {}
    for row in rows:
        groups.setdefault(row["group_id"], set()).add(row["split"])
        videos.setdefault(row["source_video_id"], set()).add(row["split"])
        if row["sha256"]:
            hashes.setdefault(row["sha256"].lower(), set()).add(row["split"])
    if any(len(value) > 1 for value in groups.values()):
        raise ValueError("split leakage: group_id occurs across train/val/test")
    if any(len(value) > 1 for value in videos.values()):
        raise ValueError("split leakage: source_video_id occurs across train/val/test")
    if any(len(value) > 1 for value in hashes.values()):
        raise ValueError("split leakage: identical SHA-256 occurs across train/val/test")
    if any(not any(row["split"] == split for row in rows) for split in ("train", "val", "test")):
        raise ValueError("train, val and test must all be non-empty")


class TranslationPairs(Dataset):
    def __init__(self, pairs, vocabulary, max_frames, max_tokens):
        self.pairs = pairs
        self.vocabulary, self.max_frames, self.max_tokens = vocabulary, max_frames, max_tokens

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        row, text = self.pairs[index]
        with np.load(row["cache"], allow_pickle=False) as cache:
            raw = cache["raw"]
        features = resample(normalize(raw), self.max_frames)
        return torch.from_numpy(features), torch.tensor(encode(text, self.vocabulary, self.max_tokens))


def collate(batch, pad_id):
    sources, targets = zip(*batch)
    source = torch.stack(sources)
    max_length = max(len(t) for t in targets)
    padded = torch.full((len(targets), max_length), pad_id, dtype=torch.long)
    for i, target in enumerate(targets):
        padded[i, :len(target)] = target
    return source, padded[:, :-1], padded[:, 1:]


def run_epoch(model, loader, optimizer, device, loss_fn, train):
    model.train(train)
    total, tokens = 0.0, 0
    context = torch.enable_grad() if train else torch.inference_mode()
    with context:
        for source, target_in, target_out in loader:
            source, target_in, target_out = source.to(device), target_in.to(device), target_out.to(device)
            logits = model(source, target_in)
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), target_out.reshape(-1))
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            count = int(target_out.ne(loss_fn.ignore_index).sum())
            total += float(loss) * count
            tokens += count
    return total / max(tokens, 1)


def generate(model, dataset, rows, refs_by_id, vocabulary, device, batch_size=4):
    model.eval()
    hypotheses, references = [], []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start:start + batch_size]
            frames = torch.stack([dataset[i][0] for i in range(start, start + len(batch_rows))]).to(device)
            out = model.generate(frames, vocabulary[BOS], vocabulary[EOS]).cpu().tolist()
            hypotheses.extend(decode(ids, vocabulary) for ids in out)
            references.extend(refs_by_id[row["sample_id"]] for row in batch_rows)
    return corpus_metrics(references, hypotheses), hypotheses


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-frames", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=3)
    args = parser.parse_args(argv)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    rows = load_rows(args.index)
    validate_splits(rows)
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    test_rows = [row for row in rows if row["split"] == "test"]
    refs_by_id = {row["sample_id"]: [str(r) for r in json.loads(row["reference_texts_json"]) if str(r).strip()] for row in rows}
    vocabulary = build_vocabulary((ref for row in train_rows for ref in refs_by_id[row["sample_id"]]),
                                  max_size=30000)
    train_pairs = [(row, ref) for row in train_rows for ref in refs_by_id[row["sample_id"]]]
    val_pairs = [(row, refs_by_id[row["sample_id"]][0]) for row in val_rows]
    test_pairs = [(row, refs_by_id[row["sample_id"]][0]) for row in test_rows]
    train_set = TranslationPairs(train_pairs, vocabulary, args.max_frames, args.max_tokens)
    val_set = TranslationPairs(val_pairs, vocabulary, args.max_frames, args.max_tokens)
    test_set = TranslationPairs(test_pairs, vocabulary, args.max_frames, args.max_tokens)
    loader = lambda data, shuffle: DataLoader(data, batch_size=args.batch_size, shuffle=shuffle,
                                               num_workers=0, collate_fn=lambda b: collate(b, vocabulary[PAD]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VideoToTextTransformer(FEATURE_DIM, len(vocabulary), dimension=args.dimension,
                                   heads=args.heads, encoder_layers=args.layers,
                                   decoder_layers=args.layers, max_frames=args.max_frames,
                                   max_tokens=args.max_tokens, pad_id=vocabulary[PAD]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocabulary[PAD], label_smoothing=0.1)
    os.makedirs(args.out, exist_ok=True)
    started = time.time(); best_val, best_epoch, bad_epochs = float("inf"), -1, 0
    for epoch in range(args.epochs):
        train_loss = run_epoch(model, loader(train_set, True), optimizer, device, loss_fn, True)
        val_loss = run_epoch(model, loader(val_set, False), optimizer, device, loss_fn, False)
        checkpoint = {"state_dict": model.state_dict(), "vocabulary": vocabulary,
                      "model_config": {"input_dim": FEATURE_DIM, "vocabulary_size": len(vocabulary),
                                       "dimension": args.dimension, "heads": args.heads,
                                       "encoder_layers": args.layers, "decoder_layers": args.layers,
                                       "max_frames": args.max_frames, "max_tokens": args.max_tokens,
                                       "pad_id": vocabulary[PAD]}, "epoch": epoch, "seed": args.seed}
        torch.save(checkpoint, os.path.join(args.out, "last.pt"))
        if val_loss < best_val:
            best_val, best_epoch, bad_epochs = val_loss, epoch, 0
            torch.save(checkpoint, os.path.join(args.out, "best.pt"))
        else:
            bad_epochs += 1
        print(f"epoch={epoch + 1} train_token_loss={train_loss:.4f} val_token_loss={val_loss:.4f} best={best_val:.4f}")
        if bad_epochs >= 8:
            break
    best = torch.load(os.path.join(args.out, "best.pt"), map_location=device, weights_only=False)
    model.load_state_dict(best["state_dict"])
    test_metrics, hypotheses = generate(model, test_set, test_rows, refs_by_id, vocabulary, device)
    with open(args.index, "rb") as stream:
        index_hash = hashlib.sha256(stream.read()).hexdigest()
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                           stderr=subprocess.DEVNULL).strip()
    except Exception:
        revision = None
    version = f"continuous-pose-v1-{index_hash[:12]}"
    best["model_version"] = version
    torch.save(best, os.path.join(args.out, "best.pt"))
    result = {"schema_version": "continuous-experiment-v1", "task": "continuous",
              "model_version": version, "source_index_sha256": index_hash,
              "git_revision": revision, "split_rows": {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)},
              "split_groups": {s: len({r["group_id"] for r in rows if r["split"] == s}) for s in ("train", "val", "test")},
              "vocabulary_size": len(vocabulary), "best_epoch": best_epoch,
              "best_validation_token_loss": best_val, "test_metrics": test_metrics,
              "test_sample_ids": [row["sample_id"] for row in test_rows],
              "test_hypotheses": hypotheses, "seed": args.seed,
              "elapsed_seconds": time.time() - started, "python": platform.python_version(),
              "torch": torch.__version__, "device": device,
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              "limitations": ["170D pose/hand features omit direct RGB, face and non-manual cues",
                              "whitespace vocabulary is an experimental baseline; no subword morphology"]}
    with open(os.path.join(args.out, "experiment.json"), "w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out, "vocabulary.json"), "w", encoding="utf-8") as stream:
        json.dump(vocabulary, stream, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out, "MODEL_CARD.md"), "w", encoding="utf-8") as stream:
        stream.write("# Experimental continuous ISL translation baseline\n\n")
        stream.write("Pose/hand-only research artifact. It is not approved for production and may omit "
                     "linguistically meaningful facial/non-manual information.\n\n")
        stream.write(f"Model version: `{version}`\n\nTest metrics: `{json.dumps(test_metrics, sort_keys=True)}`\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
