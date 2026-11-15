"""Training-only whitespace tokenizer helpers for a small research baseline."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Iterable, Mapping


PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIALS = (PAD, BOS, EOS, UNK)


def normalize_for_model(text: str) -> str:
    """Normalize model input without overwriting original manifest references."""
    value = unicodedata.normalize("NFKC", str(text)).casefold()
    return re.sub(r"\s+", " ", value).strip()


def build_vocabulary(training_references: Iterable[str], min_frequency: int = 1,
                     max_size: int = 30000) -> dict[str, int]:
    if min_frequency < 1 or max_size < len(SPECIALS):
        raise ValueError("invalid vocabulary limits")
    counts: Counter[str] = Counter()
    for sentence in training_references:
        counts.update(normalize_for_model(sentence).split())
    ordered = sorted((word for word, count in counts.items() if count >= min_frequency),
                     key=lambda word: (-counts[word], word))
    ordered = ordered[:max_size - len(SPECIALS)]
    return {token: idx for idx, token in enumerate((*SPECIALS, *ordered))}


def encode(text: str, vocabulary: Mapping[str, int], max_tokens: int = 100) -> list[int]:
    if max_tokens < 2:
        raise ValueError("max_tokens must allow BOS and EOS")
    unk_id = vocabulary[UNK]
    words = normalize_for_model(text).split()[:max_tokens - 2]
    return [vocabulary[BOS], *(vocabulary.get(word, unk_id) for word in words), vocabulary[EOS]]


def decode(ids: Iterable[int], vocabulary: Mapping[str, int]) -> str:
    inverse = {int(idx): token for token, idx in vocabulary.items()}
    words = []
    for idx in ids:
        token = inverse.get(int(idx), UNK)
        if token == EOS:
            break
        if token not in SPECIALS:
            words.append(token)
    return " ".join(words)
