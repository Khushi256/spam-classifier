"""
Phase 5 — FastAPI Deployment (with UI)
Loads the best saved model and serves predictions via a REST API.
Also serves the frontend UI from the /static folder.

Run with:
    uvicorn api.app:app --reload --port 8000

Then open: http://localhost:8000
"""

import torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from data.preprocess import tokenize, pad_sequence, Vocabulary


# ── 1. App Setup ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="SMS Spam Classifier",
    description="Classifies a text message as spam or ham using a PyTorch model.",
    version="1.0.0",
)

# Allow the browser to call the API from the same origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve everything inside /static as static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── 2. Load Model at Startup ──────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(checkpoint_path: str):
    torch.serialization.add_safe_globals([Vocabulary])

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False
    )

    model_name = checkpoint["model_name"]
    vocab      = checkpoint["vocab"]
    max_len    = checkpoint["max_len"]

    if model_name == "lstm":
        from models.architectures import LSTMClassifier
        model = LSTMClassifier(vocab_size=len(vocab)).to(DEVICE)
    else:
        from models.architectures import CNNClassifier
        model = CNNClassifier(vocab_size=len(vocab)).to(DEVICE)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print(f"[API] Loaded {model_name.upper()} | vocab={len(vocab)} | max_len={max_len}")
    print(f"[API] Best val metrics: {checkpoint['val_metrics']}")
    return model, vocab, max_len


MODEL, VOCAB, MAX_LEN = load_model("models/lstm_best.pt")


# ── 3. Schemas ────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    text: str
    label: str
    confidence: float
    is_spam: bool


# ── 4. Serve UI ───────────────────────────────────────────────────────────────
@app.get("/")
def serve_ui():
    """Serve the frontend HTML page."""
    return FileResponse("static/index.html")


# ── 5. Predict ────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_text(text: str):
    tokens  = tokenize(text)
    indices = VOCAB.encode(tokens)
    padded  = pad_sequence(indices, MAX_LEN)
    tensor  = torch.tensor([padded], dtype=torch.long).to(DEVICE)
    logit   = MODEL(tensor)
    prob    = torch.sigmoid(logit).item()
    return prob


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    if len(text) > 500:
        raise HTTPException(status_code=400, detail="Text too long (max 500 chars).")

    spam_prob = predict_text(text)
    is_spam   = spam_prob >= 0.5

    return PredictResponse(
        text=text,
        label="spam" if is_spam else "ham",
        confidence=round(spam_prob, 4),
        is_spam=is_spam,
    )


@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE)}