from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch, numpy as np, joblib, sys, os
sys.path.append(os.path.dirname(__file__))
from transformers import AutoTokenizer
from model import MultimodalSentimentModel
from feature_engineering import extract_features

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

LABEL_NAMES = ["sadness","joy","love","anger","fear","surprise"]
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
scaler    = joblib.load("models/scaler.pkl")
model     = MultimodalSentimentModel(use_lora=True)
model.load_state_dict(torch.load("models/best_bert_model.pt", map_location=DEVICE))
model.to(DEVICE); model.eval()
print("Ready!")

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(inp: TextInput):
    enc = tokenizer(inp.text, return_tensors="pt",
                    truncation=True, max_length=128)
    ids  = enc["input_ids"].to(DEVICE)
    mask = enc["attention_mask"].to(DEVICE)
    sf   = torch.tensor(
        scaler.transform(extract_features(inp.text).reshape(1,-1)),
        dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(ids, mask, sf), dim=1).cpu().numpy()[0]
    return {
        "prediction": LABEL_NAMES[int(np.argmax(probs))],
        "confidence": float(np.max(probs)),
        "scores": {name: round(float(p), 4)
                   for name, p in zip(LABEL_NAMES, probs)}
    }