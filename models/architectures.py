"""
Phase 2 — Model Architectures
Two models for comparison:
  - LSTMClassifier  : captures sequential/order context
  - CNNClassifier   : captures local n-gram patterns
Both output a single logit (use with BCEWithLogitsLoss).
"""

import torch
import torch.nn as nn


# ── 1. LSTM Classifier ────────────────────────────────────────────────────────
class LSTMClassifier(nn.Module):
    """
    Embedding → BiLSTM → take final hidden state → FC → logit

    Why BiLSTM?
      A bidirectional LSTM reads the sentence left-to-right AND
      right-to-left, so the final hidden state sees the full context.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        super().__init__()

        # Embedding: maps each word index to a dense vector
        # padding_idx=0 means the PAD token always gets a zero vector
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # BiLSTM: processes the sequence
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,       # input shape: (batch, seq_len, embed_dim)
            bidirectional=True,     # doubles the output hidden size
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        # hidden_dim * 2 because bidirectional concatenates forward + backward
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.dropout(self.embedding(x))          # (batch, seq, embed)
        _, (hidden, _) = self.lstm(embedded)                # hidden: (layers*2, batch, hidden)

        # Grab the last layer's forward and backward hidden states
        # hidden[-2] = last layer forward, hidden[-1] = last layer backward
        last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # (batch, hidden*2)
        out = self.fc(self.dropout(last_hidden))                  # (batch, 1)
        return out.squeeze(1)                                     # (batch,)


# ── 2. CNN Classifier ─────────────────────────────────────────────────────────
class CNNClassifier(nn.Module):
    """
    Embedding → parallel Conv1d filters (bigrams, trigrams, 4-grams)
    → MaxPool over time → concat → FC → logit

    Why CNN for text?
      Each filter acts like an n-gram detector.
      Max-pooling picks the strongest signal regardless of position.
      Much faster than LSTM; surprisingly competitive on short texts.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        num_filters: int = 128,
        kernel_sizes: list = None,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [2, 3, 4]   # bigrams, trigrams, 4-grams

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # One Conv1d per kernel size — they run in parallel
        # Conv1d expects (batch, channels, length)
        # in_channels = embed_dim (each position is a vector of size embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k)
            for k in kernel_sizes
        ])

        self.dropout = nn.Dropout(dropout)

        # Concat all filter outputs → single FC
        self.fc = nn.Linear(num_filters * len(kernel_sizes), 1)

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)                # (batch, seq_len, embed_dim)
        embedded = embedded.permute(0, 2, 1)        # (batch, embed_dim, seq_len) for Conv1d

        pooled = []
        for conv in self.convs:
            c = torch.relu(conv(embedded))          # (batch, num_filters, seq_len - k + 1)
            c = c.max(dim=2).values                 # (batch, num_filters) — global max pool
            pooled.append(c)

        cat = torch.cat(pooled, dim=1)              # (batch, num_filters * num_kernels)
        out = self.fc(self.dropout(cat))            # (batch, 1)
        return out.squeeze(1)                       # (batch,)


# ── 3. Quick sanity check ─────────────────────────────────────────────────────
if __name__ == "__main__":
    VOCAB = 5000
    B, L = 8, 64  # batch=8, seq_len=64

    x = torch.randint(0, VOCAB, (B, L))

    lstm_model = LSTMClassifier(VOCAB)
    cnn_model = CNNClassifier(VOCAB)

    print("LSTM output shape:", lstm_model(x).shape)   # should be (8,)
    print("CNN  output shape:", cnn_model(x).shape)    # should be (8,)
    print("Both models OK ✓")
