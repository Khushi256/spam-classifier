# SMS Spam Classifier — Pure PyTorch + FastAPI

A text classification project built **entirely from scratch** using PyTorch.
No scikit-learn. No HuggingFace. Just raw DL.

## Architecture

```
Raw Text → Tokenizer → Vocabulary → Padded Tensor
              ↓
         Embedding Layer
              ↓
    [LSTM]          [CNN]
  BiLSTM x2      Conv1d (2,3,4-gram)
  Final hidden    MaxPool over time
              ↓
         Fully Connected
              ↓
       BCEWithLogitsLoss
              ↓
         FastAPI /predict
```

## Dataset

UCI SMS Spam Collection — 5,574 messages (~87% ham, ~13% spam)
Download: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
Save as: `data/SMSSpamCollection`

## Project Structure

```
spam_classifier/
│
├── data/
│   ├── preprocess.py       # Tokenizer, Vocabulary, SMSDataset, DataLoader
│   └── SMSSpamCollection   # ← place downloaded dataset here
│
├── models/
│   ├── architectures.py    # LSTMClassifier + CNNClassifier
│   ├── lstm_best.pt        # saved after training
│   └── cnn_best.pt         # saved after training
│
├── api/
│   └── app.py              # FastAPI server
│
├── train.py                # Training loop + evaluation metrics
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
# Train both models and compare
python train.py
```

Expected output after 10 epochs:
```
LSTM  Best Val F1 = ~0.97
CNN   Best Val F1 = ~0.96
Winner: LSTM 🏆
```

## Running the API

```bash
uvicorn api.app:app --reload --port 8000
```

## Testing the API

**With curl:**
```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "WINNER! Claim your free prize now by calling 08000930705"}'
```

**Response:**
```json
{
  "text": "WINNER! Claim your free prize now by calling 08000930705",
  "label": "spam",
  "confidence": 0.9821,
  "is_spam": true
}
```

**With the auto-generated Swagger UI:**
Open http://localhost:8000/docs in your browser.

## What I Learnt From This Project

| Concept | Where it happens |
|---|---|
| Tokenization from scratch | `data/preprocess.py` |
| Vocabulary + UNK/PAD tokens | `Vocabulary` class |
| Padding sequences manually | `pad_sequence()` |
| Embedding layers | `architectures.py` |
| BiLSTM for sequences | `LSTMClassifier` |
| Conv1d for n-gram detection | `CNNClassifier` |
| BCEWithLogitsLoss + pos_weight | `train.py` |
| Why accuracy fails on imbalanced data | `compute_metrics()` |
| Model checkpointing | `train.py` |
| Serving ML via REST API | `api/app.py` |

