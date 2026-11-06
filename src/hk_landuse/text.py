from __future__ import annotations

import re
from collections import Counter


TOKEN_RE = re.compile(r"[a-z0-9_]+(?:=[a-z0-9_:+.-]+)?|[;]", re.IGNORECASE)
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def build_vocab(texts: list[str], min_frequency: int = 1) -> dict[str, int]:
    counts = Counter(token for text in texts for token in tokenize(text))
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if count >= min_frequency:
            vocab[token] = len(vocab)
    return vocab


def encode_text(text: str, vocab: dict[str, int], max_tokens: int) -> tuple[list[int], list[bool]]:
    tokens = tokenize(text)[:max_tokens]
    ids = [vocab.get(token, vocab[UNK_TOKEN]) for token in tokens]
    mask = [True] * len(ids)
    if not ids:
        ids = [vocab[UNK_TOKEN]]
        mask = [True]
    padding = max_tokens - len(ids)
    return ids + [vocab[PAD_TOKEN]] * padding, mask + [False] * padding

