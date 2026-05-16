import os
import sys
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")   # no display needed — saves to file
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import joblib

sys.path.append(os.path.dirname(__file__))
from model import MultimodalSentimentModel

# ── Config ────────────────────────────────────────────────
LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]
MODEL_NAME  = "bert-base-uncased"
DEVICE      = torch.device(
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else "cpu"
)
os.makedirs("outputs", exist_ok=True)

# ── Load model + data ─────────────────────────────────────
print("Loading model and data...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
scaler    = joblib.load("models/scaler.pkl")

model = MultimodalSentimentModel(
    model_name=MODEL_NAME, num_labels=6,
    structured_feat_dim=10, use_lora=True
)
model.load_state_dict(
    torch.load("models/best_bert_model.pt", map_location=DEVICE)
)
model.to(DEVICE)
model.eval()

dataset = load_dataset("dair-ai/emotion")

# Full evaluation: precision, recall, F1, AUC, confusion matrix
def tokenize_fn(examples):
    return tokenizer(examples["text"], padding=False,
                     truncation=True, max_length=128)

tok_ds = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
tok_ds.set_format("torch")
collator    = DataCollatorWithPadding(tokenizer)
test_loader = DataLoader(tok_ds["test"], batch_size=32,
                         shuffle=False, collate_fn=collator)

test_feats = torch.tensor(
    np.load("data/test_features.npy"), dtype=torch.float32
)

# ── Run inference on full test set ────────────────────────
print("Running inference on 2000 test samples...")
all_preds, all_labels, all_probs = [], [], []

with torch.no_grad():
    for step, batch in enumerate(test_loader):
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)

        bs      = input_ids.size(0)
        start   = step * 32
        s_feats = test_feats[start:start + bs].to(DEVICE)

        logits = model(input_ids, attention_mask, s_feats)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = np.argmax(probs, axis=1)

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

all_probs  = np.array(all_probs)
all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

# ── Metrics ───────────────────────────────────────────────
acc  = accuracy_score(all_labels, all_preds)
f1_w = f1_score(all_labels, all_preds, average="weighted")
f1_m = f1_score(all_labels, all_preds, average="macro")
prec = precision_score(all_labels, all_preds,
                       average="weighted", zero_division=0)
rec  = recall_score(all_labels, all_preds,
                    average="weighted", zero_division=0)

# AUC — one-vs-rest
labels_bin = label_binarize(all_labels, classes=list(range(6)))
auc = roc_auc_score(labels_bin, all_probs,
                    average="weighted", multi_class="ovr")

print("\n" + "="*55)
print("FINAL EVALUATION — Fine-tuned BERT (Full Test Set)")
print("="*55)
print(f"  Accuracy           : {acc:.4f}")
print(f"  Precision (weighted): {prec:.4f}")
print(f"  Recall    (weighted): {rec:.4f}")
print(f"  F1        (weighted): {f1_w:.4f}")
print(f"  F1        (macro)   : {f1_m:.4f}")
print(f"  AUC       (weighted): {auc:.4f}")
print("="*55)
print("\nPer-class report:")
print(classification_report(all_labels, all_preds,
                             target_names=LABEL_NAMES))

# ── Plot 1: Confusion matrix ──────────────────────────────
print("Generating plots...")
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
plt.colorbar(im, ax=ax)
ax.set_xticks(range(6)); ax.set_yticks(range(6))
ax.set_xticklabels(LABEL_NAMES, rotation=45, ha="right")
ax.set_yticklabels(LABEL_NAMES)
for i in range(6):
    for j in range(6):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > cm.max()/2 else "black",
                fontsize=11)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title("Confusion Matrix — Fine-tuned BERT")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=150)
plt.close()

# ── Plot 2: Per-class F1 bar chart ────────────────────────
f1_per_class = f1_score(all_labels, all_preds,
                         average=None, zero_division=0)
colors = ["#4C9BE8","#56C596","#F4845F",
          "#E8C94C","#A78BFA","#F472B6"]
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(LABEL_NAMES, f1_per_class, color=colors, edgecolor="none")
for bar, val in zip(bars, f1_per_class):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f"{val:.2f}", ha="center", va="bottom", fontsize=11)
ax.set_ylim(0, 1.1)
ax.set_ylabel("F1 Score")
ax.set_title("Per-class F1 Score — Fine-tuned BERT")
ax.axhline(y=f1_w, color="gray", linestyle="--",
           linewidth=1, label=f"Weighted avg: {f1_w:.2f}")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/per_class_f1.png", dpi=150)
plt.close()

# ── Plot 3: Model comparison bar chart ───────────────────
methods  = ["Zero-shot\n(Mistral)", "Few-shot\n(Mistral)",
            "Fine-tuned\nBERT (ours)"]
f1_scores = [0.549, 0.539, f1_w]
bar_colors = ["#94A3B8", "#94A3B8", "#4C9BE8"]
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(methods, f1_scores, color=bar_colors,
              width=0.5, edgecolor="none")
for bar, val in zip(bars, f1_scores):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", va="bottom",
            fontsize=12, fontweight="bold")
ax.set_ylim(0, 1.1)
ax.set_ylabel("Weighted F1 Score")
ax.set_title("Model Comparison: Fine-tuned vs Baselines")
plt.tight_layout()
plt.savefig("outputs/model_comparison.png", dpi=150)
plt.close()

# ── Plot 4: ROC curves ───────────────────────────────────
from sklearn.metrics import roc_curve, auc as auc_score
fig, ax = plt.subplots(figsize=(8, 6))
for i, (name, color) in enumerate(zip(LABEL_NAMES, colors)):
    fpr, tpr, _ = roc_curve(labels_bin[:, i], all_probs[:, i])
    roc_auc     = auc_score(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2,
            label=f"{name} (AUC = {roc_auc:.2f})")
ax.plot([0,1],[0,1],"k--", lw=1)
ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — Fine-tuned BERT (one-vs-rest)")
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig("outputs/roc_curves.png", dpi=150)
plt.close()

# ── Save final report ─────────────────────────────────────
report = {
    "model": "Fine-tuned BERT + LoRA + Cross-modal Fusion",
    "test_samples": len(all_labels),
    "metrics": {
        "accuracy":           round(acc,  4),
        "precision_weighted": round(prec, 4),
        "recall_weighted":    round(rec,  4),
        "f1_weighted":        round(f1_w, 4),
        "f1_macro":           round(f1_m, 4),
        "auc_weighted":       round(auc,  4),
    },
    "baseline_comparison": {
        "zero_shot_f1": 0.549,
        "few_shot_f1":  0.539,
        "fine_tuned_f1": round(f1_w, 4),
        "improvement_over_zero_shot":
            round((f1_w - 0.549) / 0.549 * 100, 1)
    }
}
with open("outputs/final_report.json", "w") as f:
    json.dump(report, f, indent=2)

print("\nAll outputs saved to outputs/:")
print("  confusion_matrix.png")
print("  per_class_f1.png")
print("  model_comparison.png")
print("  roc_curves.png")
print("  final_report.json")
print(f"\nImprovement over zero-shot baseline: "
      f"+{report['baseline_comparison']['improvement_over_zero_shot']}%")
print("\nPhase 7 complete — project finished!")