# Multimodal Sentiment & Intent Classifier

A production-grade NLP system combining fine-tuned BERT/RoBERTa with structured 
feature signals to classify emotion and intent across 6 categories with **91.6% accuracy**.

## Results

| Model | Accuracy | F1 Score | AUC |
|-------|----------|----------|-----|
| Zero-shot Mistral (baseline) | 55.0% | 0.549 | — |
| Few-shot Mistral (baseline) | 52.0% | 0.539 | — |
| **Fine-tuned BERT + LoRA (ours)** | **91.6%** | **0.916** | **0.993** |

**+66.8% improvement over zero-shot LLM baseline.**

## Architecture

```
Raw Text
   │
   ├──► BERT (bert-base-uncased)
   │         + LoRA adapters (0.27% params trained)
   │         → [CLS] embedding (768-dim)
   │
   ├──► Structured Feature Extractor
   │         → 10 numeric signals (word count, caps ratio, etc.)
   │         → StandardScaler normalisation
   │
   └──► Cross-modal Fusion
             torch.cat([768-dim, 10-dim]) → 778-dim vector
             → Linear(778, 256) → ReLU → Dropout → Linear(256, 6)
             → Predicted emotion
```

## Emotion Classes
`sadness` · `joy` · `love` · `anger` · `fear` · `surprise`

## Tech Stack
- **PyTorch** — training backbone
- **HuggingFace Transformers** — BERT, RoBERTa, tokenizers
- **PEFT** — LoRA adapter fine-tuning
- **scikit-learn** — metrics, StandardScaler
- **Ollama + Mistral** — free LLM baseline comparison
- **React + FastAPI** — interactive demo

## Project Structure
```
├── src/
│   ├── data_pipeline.py       # tokenization, dynamic padding, augmentation
│   ├── feature_engineering.py # 10 structured features + normalisation
│   ├── model.py               # BERT + LoRA + fusion architecture
│   ├── train.py               # training loop with warmup scheduler
│   ├── predict.py             # single-sentence inference
│   ├── baseline.py            # zero-shot & few-shot LLM baselines
│   └── evaluate.py            # metrics, confusion matrix, ROC curves
├── data/                      # processed feature arrays
├── models/                    # saved weights + scaler
└── outputs/                   # evaluation charts + final report
```

## Setup
```bash
git clone https://github.com/YOUR_USERNAME/multimodal-sentiment-classifier
cd multimodal-sentiment-classifier
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Training
```bash
python src/feature_engineering.py   # build structured features
python src/train.py                  # fine-tune BERT with LoRA
python src/evaluate.py               # generate metrics + charts
```

## Key Design Decisions
- **LoRA over full fine-tuning** — trains only 294K of 110M parameters, 
  making it feasible on a laptop CPU/MPS without quality loss
- **Dynamic padding** — pads each batch to its longest sequence, 
  not a fixed 512 — saves ~70% memory during training
- **Cross-modal fusion** — concatenating structured signals with BERT 
  embeddings improves robustness on underrepresented classes
- **Scaler fitted on train only** — prevents data leakage from val/test sets

## Dataset
[dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) — 20,000 English Twitter messages labelled with 6 emotions (sadness, joy, love, anger, fear, surprise). Loaded via HuggingFace datasets library.
