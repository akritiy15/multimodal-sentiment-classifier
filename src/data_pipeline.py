from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from collections import Counter
import pandas as pd
import nlpaug.augmenter.word as naw
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
from deep_translator import GoogleTranslator

# Step 2 — Load dataset
dataset = load_dataset("dair-ai/emotion")
print(dataset)
print(dataset["train"][0])

# Step 3 — Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
sample = "I feel absolutely wonderful today!"
tokens = tokenizer(sample, return_tensors="pt")
print("\nSample tokens:", tokens)

# Step 4 — Tokenize full dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding=False,
        truncation=True,
        max_length=128
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)
tokenized_dataset.set_format("torch")
print("\nTokenized sample:", tokenized_dataset["train"][0])

# Step 5 — DataLoader with dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_loader = DataLoader(
    tokenized_dataset["train"],
    batch_size=16,
    shuffle=True,
    collate_fn=data_collator
)
val_loader = DataLoader(
    tokenized_dataset["validation"],
    batch_size=16,
    shuffle=False,
    collate_fn=data_collator
)

batch = next(iter(train_loader))
print("\ninput_ids shape:", batch["input_ids"].shape)
print("attention_mask shape:", batch["attention_mask"].shape)
print("labels shape:", batch["labels"].shape)

# Step 6 — Class distribution
label_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]
train_labels = [int(x) for x in tokenized_dataset["train"]["label"]]
counts = Counter(train_labels)
df = pd.DataFrame([
    {"emotion": label_names[k], "count": v}
    for k, v in sorted(counts.items())
])
print("\nClass distribution:")
print(df)

# Step 7 — Synonym augmentation
aug = naw.SynonymAug(aug_src="wordnet")

def augment_texts(texts, n=1):
    augmented = []
    for text in texts:
        try:
            results = aug.augment(text, n=n)
            augmented.extend(results)
        except Exception:
            augmented.append(text)
    return augmented

love_texts = [
    dataset["train"][i]["text"]
    for i in range(len(dataset["train"]))
    if dataset["train"][i]["label"] == 2
]
augmented_love = augment_texts(love_texts[:5])
print("\nSynonym augmentation:")
for orig, aug_text in zip(love_texts[:5], augmented_love):
    print(f"  Original:   {orig}")
    print(f"  Augmented:  {aug_text}\n")

# Step 8 — Back-translation
def back_translate(text, mid_lang="fr"):
    try:
        translated = GoogleTranslator(source="en", target=mid_lang).translate(text)
        back = GoogleTranslator(source=mid_lang, target="en").translate(translated)
        return back
    except Exception:
        return text

original = "I feel so scared and anxious about tomorrow"
result = back_translate(original)
print("Back-translation:")
print(f"  Original:        {original}")
print(f"  Back-translated: {result}")