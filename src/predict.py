import torch
import numpy as np
import joblib
import sys
import os
sys.path.append(os.path.dirname(__file__))

from transformers import AutoTokenizer
from model import MultimodalSentimentModel
from feature_engineering import extract_features

LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]
MODEL_NAME  = "bert-base-uncased"
MODEL_PATH  = "models/best_bert_model.pt"
SCALER_PATH = "models/scaler.pkl"

DEVICE = torch.device(
    "mps"  if torch.backends.mps.is_available()  else
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_model():
    """Load the saved fine-tuned model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    scaler    = joblib.load(SCALER_PATH)

    model = MultimodalSentimentModel(
        model_name=MODEL_NAME,
        num_labels=6,
        structured_feat_dim=10,
        use_lora=True
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model, tokenizer, scaler


def predict(text: str, model, tokenizer, scaler):
    """Run full multimodal prediction on a single sentence."""

    # 1 — Tokenize text
    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    )
    input_ids      = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    # 2 — Extract + scale structured features
    raw_feats    = extract_features(text).reshape(1, -1)   # shape (1, 10)
    scaled_feats = scaler.transform(raw_feats)             # normalise
    s_feats      = torch.tensor(scaled_feats,
                                dtype=torch.float32).to(DEVICE)

    # 3 — Forward pass through fused model
    with torch.no_grad():
        logits = model(input_ids, attention_mask, s_feats)

    # 4 — Convert logits to probabilities
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx   = int(np.argmax(probs))
    pred_label = LABEL_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    return pred_label, confidence, probs


def show_result(text, model, tokenizer, scaler):
    """Pretty-print prediction with all class probabilities."""
    label, confidence, probs = predict(text, model, tokenizer, scaler)

    print(f"\nText      : {text}")
    print(f"Prediction: {label.upper()} ({confidence*100:.1f}% confident)")
    print("All scores:")
    for name, prob in sorted(zip(LABEL_NAMES, probs),
                              key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"  {name:10s} {prob*100:5.1f}%  {bar}")


if __name__ == "__main__":
    print("Loading model...")
    model, tokenizer, scaler = load_model()
    print("Model loaded!\n")

    test_sentences = [
        "I am so happy and excited about my new job!",
        "I feel terrified and cannot stop shaking.",
        "This is the worst day of my life, I hate everything.",
        "I love you so much, you mean the world to me.",
        "I am completely shocked, I never expected this.",
        "I feel empty and hopeless inside.",
    ]

    for sentence in test_sentences:
        show_result(sentence, model, tokenizer, scaler)