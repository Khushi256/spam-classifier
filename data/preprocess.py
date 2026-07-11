"""
Phase 1 — Data & Preprocessing
Dataset: UCI SMS Spam Collection
No sklearn. Pure Python + PyTorch.
"""

import re    # for regex-based tokenization (regular expression) -text cleaning
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter  # for counting token frequencies to build the vocabulary


# ── 1. Tokenizer ──────────────────────────────────────────────────────────────
def tokenize(text: str) -> list[str]:
    """Lowercase, remove punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip().split()


# ── 2. Vocabulary (NLP concept) ─────────────────────────────────────────────────────────────
class Vocabulary:
    """Build word → index mapping from a list of tokenized sentences."""

    PAD = "<PAD>"   # index 0  — used for padding shorter sequences
    UNK = "<UNK>"   # index 1  — used for words not seen during training

    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.word2idx = {self.PAD: 0, self.UNK: 1}
        self.idx2word = {0: self.PAD, 1: self.UNK} #used for debugging

    def build(self, tokenized_texts: list[list[str]]):
        """Count every token; add those that meet min_freq threshold."""
        counter = Counter(token for tokens in tokenized_texts for token in tokens)
        for word, freq in counter.items():
            if freq >= self.min_freq:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        print(f"[Vocab] size = {len(self.word2idx)} tokens")

    def encode(self, tokens: list[str]) -> list[int]:
        """Convert tokens → indices (unknown words map to UNK)."""
        return [self.word2idx.get(t, 1) for t in tokens]

    def __len__(self):
        return len(self.word2idx)


# ── 3. Padding helper ─────────────────────────────────────────────────────────
def pad_sequence(indices: list[int], max_len: int) -> list[int]:
    """Truncate if too long; pad with 0 (PAD index) if too short."""
    if len(indices) >= max_len:
        return indices[:max_len]
    return indices + [0] * (max_len - len(indices))


# ── 4. Dataset ────────────────────────────────────────────────────────────────
class SMSDataset(Dataset):
    """
    Reads the UCI SMS Spam file (tab-separated: label\\ttext).
    Download from: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
    Save as data/SMSSpamCollection (no extension, as distributed).
    """

    LABEL_MAP = {"ham": 0, "spam": 1}

    def __init__(self, filepath: str, vocab: Vocabulary, max_len: int = 64):
        self.max_len = max_len
        self.samples = []

        with open(filepath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                label_str, text = line.split("\t", 1)
                tokens = tokenize(text)
                indices = vocab.encode(tokens)
                padded = pad_sequence(indices, max_len)
                label = self.LABEL_MAP[label_str]
                self.samples.append((padded, label))

        print(f"[Dataset] loaded {len(self.samples)} samples from {filepath}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        indices, label = self.samples[idx]
        return (
            torch.tensor(indices, dtype=torch.long),
            torch.tensor(label, dtype=torch.float32),
        )


# ── 5. Build vocab + loaders (called from train.py) ──────────────────────────
def build_vocab_and_loaders(
    filepath: str,
    max_len: int = 64,
    batch_size: int = 32,
    train_split: float = 0.8,
):
    """
    Returns (vocab, train_loader, val_loader).
    Splits data manually — no sklearn train_test_split.
    """
    # First pass: build vocabulary
    raw_texts = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                _, text = line.split("\t", 1)
                raw_texts.append(tokenize(text))

    vocab = Vocabulary(min_freq=2)
    vocab.build(raw_texts)

    # Second pass: build full dataset
    dataset = SMSDataset(filepath, vocab, max_len)

    # Manual split
    n = len(dataset)
    n_train = int(n * train_split)
    n_val = n - n_train

    # Use torch's random_split (pure PyTorch, no sklearn)
    from torch.utils.data import random_split
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    print(f"[Split] train={n_train}, val={n_val}")
    return vocab, train_loader, val_loader
