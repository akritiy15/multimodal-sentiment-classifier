import re
import os
import string
import numpy as np
import pandas as pd
import joblib
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler

NEGATION_WORDS = {
    "not", "never", "no", "nobody", "nothing",
    "neither", "nor", "nowhere", "hardly", "barely", "scarcely"
}

POSITIVE_WORDS = {
    "love", "happy", "great", "wonderful", "amazing",
    "fantastic", "excellent", "good", "joy", "excited",
    "glad", "pleased", "brilliant", "awesome", "beautiful"
}

NEGATIVE_WORDS = {
    "hate", "sad", "terrible", "awful", "horrible",
    "bad", "angry", "fear", "scared", "miserable",
    "depressed", "anxious", "worried", "upset", "angry"
}

FEATURE_NAMES = [
    "text_length", "word_count", "avg_word_length",
    "exclamation_count", "question_count", "capital_ratio",
    "punct_count", "neg_word_count", "positive_word_count",
    "negative_word_count"
]


def extract_features(text: str) -> np.ndarray:
    """Extract 10 structured features from a raw text string."""
    words = text.split()
    lower_words = [w.lower().strip(string.punctuation) for w in words]

    text_length         = len(text)
    word_count          = len(words)
    avg_word_length     = np.mean([len(w) for w in words]) if words else 0.0
    exclamation_count   = text.count("!")
    question_count      = text.count("?")
    capital_ratio       = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    punct_count         = sum(1 for c in text if c in string.punctuation)
    neg_word_count      = sum(1 for w in lower_words if w in NEGATION_WORDS)
    positive_word_count = sum(1 for w in lower_words if w in POSITIVE_WORDS)
    negative_word_count = sum(1 for w in lower_words if w in NEGATIVE_WORDS)

    return np.array([
        text_length,
        word_count,
        avg_word_length,
        exclamation_count,
        question_count,
        capital_ratio,
        punct_count,
        neg_word_count,
        positive_word_count,
        negative_word_count,
    ], dtype=np.float32)


def build_feature_matrix(dataset_split) -> np.ndarray:
    """Build an (N, 10) feature matrix for an entire dataset split."""
    return np.array([
        extract_features(text) for text in dataset_split["text"]
    ], dtype=np.float32)


def normalise_features(train_feats, val_feats, test_feats,
                        save_path="models/scaler.pkl"):
    """Fit scaler on train only, apply to val and test. Save for later use."""
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_feats)
    val_scaled   = scaler.transform(val_feats)
    test_scaled  = scaler.transform(test_feats)

    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, save_path)
    print(f"Scaler saved to {save_path}")
    return train_scaled, val_scaled, test_scaled


if __name__ == "__main__":
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("dair-ai/emotion")

    # Build feature matrices
    print("Extracting features...")
    train_features = build_feature_matrix(dataset["train"])
    val_features   = build_feature_matrix(dataset["validation"])
    test_features  = build_feature_matrix(dataset["test"])

    print(f"Train feature matrix shape : {train_features.shape}")
    print(f"Val   feature matrix shape : {val_features.shape}")
    print(f"Test  feature matrix shape : {test_features.shape}")

    # Show features for first sample
    sample_text  = dataset["train"][0]["text"]
    sample_feats = extract_features(sample_text)
    print(f"\nSample text: '{sample_text}'")
    for name, val in zip(FEATURE_NAMES, sample_feats):
        print(f"  {name:25s}: {val:.4f}")

    # Normalise
    print("\nNormalising features...")
    train_scaled, val_scaled, test_scaled = normalise_features(
        train_features, val_features, test_features
    )

    print(f"\nBefore scaling — text_length mean : {train_features[:, 0].mean():.2f}")
    print(f"After  scaling — text_length mean : {train_scaled[:, 0].mean():.4f}")

    # Save to disk
    os.makedirs("data", exist_ok=True)
    np.save("data/train_features.npy", train_scaled)
    np.save("data/val_features.npy",   val_scaled)
    np.save("data/test_features.npy",  test_scaled)
    print("\nFeature arrays saved:")
    print("  data/train_features.npy")
    print("  data/val_features.npy")
    print("  data/test_features.npy")
    print("\nPhase 3 complete!")