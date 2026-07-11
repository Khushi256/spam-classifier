"""
Phase 3 & 4 — Training Loop + Evaluation
Metrics: Accuracy, Precision, Recall, F1 (all manual, no sklearn).
Saves the best model checkpoint based on val F1.
"""

import os
import torch
import torch.nn as nn
from data.preprocess import build_vocab_and_loaders
from models.architectures import LSTMClassifier, CNNClassifier


# ── 1. Manual Metrics (no sklearn) ───────────────────────────────────────────
def compute_metrics(preds: list[int], labels: list[int]) -> dict:
    """
    Binary classification metrics from scratch.
    preds and labels are plain Python lists of 0/1.
    """
    tp = sum(p == 1 and l == 1 for p, l in zip(preds, labels))
    fp = sum(p == 1 and l == 0 for p, l in zip(preds, labels))
    fn = sum(p == 0 and l == 1 for p, l in zip(preds, labels))
    tn = sum(p == 0 and l == 0 for p, l in zip(preds, labels))

    accuracy  = (tp + tn) / (tp + tn + fp + fn + 1e-9)
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)

    return {
        "accuracy":  round(accuracy,  4),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# ── 2. One Epoch of Training ──────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for texts, labels in loader:
        texts, labels = texts.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(texts)               # raw scores, not probabilities
        loss = criterion(logits, labels)    # BCEWithLogitsLoss handles sigmoid internally
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Convert logits → binary predictions (threshold = 0.5)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long().cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.long().cpu().tolist())

    avg_loss = total_loss / len(loader)
    metrics = compute_metrics(all_preds, all_labels)
    return avg_loss, metrics


# ── 3. Validation ─────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for texts, labels in loader:
        texts, labels = texts.to(device), labels.to(device)
        logits = model(texts)
        loss = criterion(logits, labels)
        total_loss += loss.item()

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long().cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.long().cpu().tolist())

    avg_loss = total_loss / len(loader)
    metrics = compute_metrics(all_preds, all_labels)
    return avg_loss, metrics


# ── 4. Full Training Run ──────────────────────────────────────────────────────
def train(model_name: str = "lstm", epochs: int = 10):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_PATH = "data/SMSSpamCollection"
    SAVE_PATH = f"models/{model_name}_best.pt"
    MAX_LEN = 64
    BATCH_SIZE = 32

    print(f"\n{'='*50}")
    print(f" Training: {model_name.upper()}  |  device: {DEVICE}")
    print(f"{'='*50}\n")

    # Build vocab and dataloaders
    vocab, train_loader, val_loader = build_vocab_and_loaders(
        DATA_PATH, max_len=MAX_LEN, batch_size=BATCH_SIZE
    )

    # Select model
    if model_name == "lstm":
        model = LSTMClassifier(vocab_size=len(vocab)).to(DEVICE)
    elif model_name == "cnn":
        model = CNNClassifier(vocab_size=len(vocab)).to(DEVICE)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Loss: BCEWithLogitsLoss is numerically stable (combines sigmoid + BCE)
    # pos_weight handles class imbalance (spam is ~13% of dataset)
    # pos_weight=6 means spam mistakes are penalized 6x more than ham mistakes
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([6.0]).to(DEVICE))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Reduce LR when val loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    best_f1 = 0.0

    for epoch in range(1, epochs + 1):
        train_loss, train_m = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_m     = evaluate(model, val_loader, criterion, DEVICE)

        scheduler.step(val_m["f1"])

        print(
            f"Epoch {epoch:02d}/{epochs}  "
            f"| Train loss={train_loss:.4f} acc={train_m['accuracy']:.4f} f1={train_m['f1']:.4f}"
            f"  | Val loss={val_loss:.4f} acc={val_m['accuracy']:.4f} "
            f"f1={val_m['f1']:.4f} prec={val_m['precision']:.4f} rec={val_m['recall']:.4f}"
        )

        # Save best checkpoint
        if val_m["f1"] > best_f1:
            best_f1 = val_m["f1"]
            torch.save({
                "model_state": model.state_dict(),
                "vocab": vocab,
                "model_name": model_name,
                "max_len": MAX_LEN,
                "val_metrics": val_m,
            }, SAVE_PATH)
            print(f"  ✓ Saved best model (f1={best_f1:.4f})")

    print(f"\nBest Val F1: {best_f1:.4f}")
    print(f"Model saved to: {SAVE_PATH}\n")
    return best_f1


# ── 5. Compare both models ────────────────────────────────────────────────────
if __name__ == "__main__":
    results = {}
    for arch in ["lstm", "cnn"]:
        f1 = train(model_name=arch, epochs=10)
        results[arch] = f1

    print("\n" + "="*40)
    print("   FINAL COMPARISON")
    print("="*40)
    for arch, f1 in results.items():
        print(f"  {arch.upper():<6} Best Val F1 = {f1:.4f}")
    winner = max(results, key=results.get)
    print(f"\n  Winner: {winner.upper()} 🏆")
    print("="*40)
