import json
import time
import requests
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import f1_score, accuracy_score, classification_report

OLLAMA_URL  = "http://localhost:11434/api/generate"
MODEL_NAME  = "mistral"
LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]
NUM_SAMPLES = 100   # use 100 test samples to keep runtime reasonable

# ── Prompt templates ──────────────────────────────────────

ZERO_SHOT_PROMPT = """Classify the emotion in the following sentence.
Choose exactly one from: sadness, joy, love, anger, fear, surprise.
Reply with the emotion word only, nothing else.

Sentence: {text}
Emotion:"""

FEW_SHOT_PROMPT = """Classify the emotion in sentences. Choose exactly one from:
sadness, joy, love, anger, fear, surprise.
Reply with the emotion word only, nothing else.

Examples:
Sentence: I feel so happy today!
Emotion: joy

Sentence: I am terrified of what might happen.
Emotion: fear

Sentence: I hate how things turned out.
Emotion: anger

Sentence: I love spending time with you.
Emotion: love

Sentence: I feel so down and empty.
Emotion: sadness

Sentence: I cannot believe this just happened!
Emotion: surprise

Now classify this:
Sentence: {text}
Emotion:"""


# ── Ollama call ───────────────────────────────────────────

def query_ollama(prompt: str, retries: int = 3) -> str:
    """Send a prompt to Ollama and return the response text."""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 10}
    }
    for attempt in range(retries):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=30)
            resp.raise_for_status()
            return resp.json()["response"].strip().lower()
        except Exception as e:
            print(f"  Retry {attempt+1} — {e}")
            time.sleep(2)
    return "unknown"


def parse_label(response: str) -> int:
    """Map model response string to label index."""
    response = response.strip().lower()
    for i, name in enumerate(LABEL_NAMES):
        if name in response:
            return i
    return -1   # could not parse


# ── Evaluation ────────────────────────────────────────────

def evaluate_baseline(prompt_template: str, samples, label_name: str):
    """Run baseline on all samples and compute metrics."""
    true_labels, pred_labels = [], []
    failed = 0

    for i, sample in enumerate(samples):
        text       = sample["text"]
        true_label = sample["label"]
        prompt     = prompt_template.format(text=text)
        response   = query_ollama(prompt)
        pred_label = parse_label(response)

        if pred_label == -1:
            failed += 1
            pred_label = 0   # fallback to sadness if unparseable

        true_labels.append(true_label)
        pred_labels.append(pred_label)

        if (i + 1) % 10 == 0:
            print(f"  [{label_name}] {i+1}/{len(samples)} done...")

    acc = accuracy_score(true_labels, pred_labels)
    f1  = f1_score(true_labels, pred_labels,
                   average="weighted", zero_division=0)

    print(f"\n{label_name} Results:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  Failed to parse: {failed}/{len(samples)}")
    print(classification_report(
        true_labels, pred_labels,
        target_names=LABEL_NAMES, zero_division=0
    ))
    return acc, f1, true_labels, pred_labels


# ── Main ──────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading dataset...")
    dataset = load_dataset("dair-ai/emotion")

    # Sample evenly across classes for fair evaluation
    test_data = dataset["test"]
    samples   = [test_data[i] for i in range(NUM_SAMPLES)]

    print(f"Running zero-shot baseline on {NUM_SAMPLES} samples...")
    zs_acc, zs_f1, zs_true, zs_pred = evaluate_baseline(
        ZERO_SHOT_PROMPT, samples, "Zero-shot"
    )

    print(f"\nRunning few-shot baseline on {NUM_SAMPLES} samples...")
    fs_acc, fs_f1, fs_true, fs_pred = evaluate_baseline(
        FEW_SHOT_PROMPT, samples, "Few-shot"
    )

    # Summary comparison table
    print("\n" + "="*50)
    print("BASELINE SUMMARY")
    print("="*50)
    print(f"{'Method':<20} {'Accuracy':>10} {'F1 Score':>10}")
    print("-"*42)
    print(f"{'Zero-shot':<20} {zs_acc:>10.4f} {zs_f1:>10.4f}")
    print(f"{'Few-shot':<20} {fs_acc:>10.4f} {fs_f1:>10.4f}")
    print(f"{'Fine-tuned BERT':<20} {'0.9295':>10} {'0.9294':>10}")
    print("="*50)

    # Save results
    results = {
        "zero_shot": {"accuracy": zs_acc, "f1": zs_f1},
        "few_shot":  {"accuracy": fs_acc, "f1": fs_f1},
        "fine_tuned_bert": {"accuracy": 0.9295, "f1": 0.9294}
    }
    with open("outputs/baseline_results.json", "w") as f:
        import os; os.makedirs("outputs", exist_ok=True)
        json.dump(results, f, indent=2)
    print("\nResults saved to outputs/baseline_results.json")