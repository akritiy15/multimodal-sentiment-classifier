import os
import torch
import numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
from sklearn.metrics import f1_score, accuracy_score
from model import MultimodalSentimentModel

DEVICE     = torch.device("mps" if torch.backends.mps.is_available()
             else "cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 6
BATCH_SIZE = 16
EPOCHS     = 3
LR         = 2e-4

print(f"Using device: {DEVICE}")

# ── Data ──────────────────────────────────────────────────
dataset   = load_dataset("dair-ai/emotion")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(examples):
    return tokenizer(examples["text"], padding=False,
                     truncation=True, max_length=128)

tok_ds = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
tok_ds.set_format("torch")

collator    = DataCollatorWithPadding(tokenizer)
train_loader = DataLoader(tok_ds["train"],      batch_size=BATCH_SIZE,
                          shuffle=True,  collate_fn=collator)
val_loader   = DataLoader(tok_ds["validation"], batch_size=BATCH_SIZE,
                          shuffle=False, collate_fn=collator)

# Load pre-computed structured features
train_feats = torch.tensor(
    np.load("data/train_features.npy"), dtype=torch.float32)
val_feats   = torch.tensor(
    np.load("data/val_features.npy"),   dtype=torch.float32)

# ── Model ─────────────────────────────────────────────────
model = MultimodalSentimentModel(
    model_name=MODEL_NAME,
    num_labels=NUM_LABELS,
    structured_feat_dim=10,
    use_lora=True
).to(DEVICE)

# ── Optimiser + Scheduler ─────────────────────────────────
optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR, weight_decay=0.01
)
total_steps   = len(train_loader) * EPOCHS
warmup_steps  = total_steps // 10
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)
loss_fn = torch.nn.CrossEntropyLoss()

# ── Train + Eval functions ────────────────────────────────
# Warmup scheduler + gradient clipping for stable LoRA training
def train_epoch(epoch):
    model.train()
    total_loss, all_preds, all_labels = 0, [], []

    for step, batch in enumerate(train_loader):
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)

        # Pull matching structured features for this batch
        indices  = list(range(
            step * BATCH_SIZE,
            min((step + 1) * BATCH_SIZE, len(train_feats))
        ))
        s_feats  = train_feats[indices].to(DEVICE)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, s_feats)
        loss   = loss_fn(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

        if (step + 1) % 100 == 0:
            print(f"  Epoch {epoch} | Step {step+1}/{len(train_loader)} "
                  f"| Loss: {total_loss/(step+1):.4f}")

    f1  = f1_score(all_labels, all_preds, average="weighted")
    acc = accuracy_score(all_labels, all_preds)
    print(f"  Train — Loss: {total_loss/len(train_loader):.4f} "
          f"| Acc: {acc:.4f} | F1: {f1:.4f}")


def evaluate():
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []

    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)

            indices = list(range(
                step * BATCH_SIZE,
                min((step + 1) * BATCH_SIZE, len(val_feats))
            ))
            s_feats = val_feats[indices].to(DEVICE)

            logits = model(input_ids, attention_mask, s_feats)
            loss   = loss_fn(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

    f1  = f1_score(all_labels, all_preds, average="weighted")
    acc = accuracy_score(all_labels, all_preds)
    print(f"  Val   — Loss: {total_loss/len(val_loader):.4f} "
          f"| Acc: {acc:.4f} | F1: {f1:.4f}")
    return f1


# ── Main training loop ────────────────────────────────────
best_f1 = 0.0
os.makedirs("models", exist_ok=True)

for epoch in range(1, EPOCHS + 1):
    print(f"\n── Epoch {epoch}/{EPOCHS} ──")
    train_epoch(epoch)
    val_f1 = evaluate()

    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), "models/best_bert_model.pt")
        print(f"  Model saved — best F1: {best_f1:.4f}")

print(f"\nTraining complete. Best validation F1: {best_f1:.4f}")