# ============================================================
#  train_model.py — Fine-tune RoBERTa-Urdu for Fake News
#  Run: python backend/model/train_model.py
#  GPU recommended (CUDA or MPS). Falls back to CPU.
#  Author : Salar Ahmed | FYP 2025-2026
# ============================================================

import os, torch, pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np

# ── Config ───────────────────────────────────────────────────
BASE_MODEL  = "urduhack/roberta-urdu-small"   # Urdu RoBERTa
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "bert_model")
DATA_DIR    = os.path.join(os.path.dirname(__file__), "../../fake_news_project")
EPOCHS      = 5
BATCH_SIZE  = 16
LR          = 2e-5
MAX_LEN     = 256
SEED        = 42

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🖥  Training on: {device.upper()}")


# ── Load & prepare data ──────────────────────────────────────
def load_data():
    true_df = pd.read_csv(os.path.join(DATA_DIR, "True.csv"))
    fake_df = pd.read_csv(os.path.join(DATA_DIR, "Fake.csv"))
    true_df["label"] = 1   # TRUE  = 1
    fake_df["label"] = 0   # FAKE  = 0

    df = pd.concat([true_df, fake_df], ignore_index=True).sample(frac=1, random_state=SEED)
    text_col = "text" if "text" in df.columns else df.columns[0]
    df = df[[text_col, "label"]].rename(columns={text_col: "text"})
    df["text"] = df["text"].fillna("").str.strip()

    train, test = train_test_split(df, test_size=0.15, random_state=SEED, stratify=df["label"])
    train, val  = train_test_split(train, test_size=0.1,  random_state=SEED, stratify=train["label"])
    print(f"📊 Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")
    return train, val, test


# ── Tokenisation ─────────────────────────────────────────────
def tokenize(tokenizer, df: pd.DataFrame) -> Dataset:
    ds = Dataset.from_pandas(df.reset_index(drop=True))
    return ds.map(
        lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=MAX_LEN),
        batched=True,
    )


# ── Metrics ──────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1"      : f1_score(labels, preds, average="weighted"),
    }


# ── Main ─────────────────────────────────────────────────────
def main():
    train_df, val_df, test_df = load_data()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model     = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)
    model.to(device)

    train_ds = tokenize(tokenizer, train_df)
    val_ds   = tokenize(tokenizer, val_df)
    test_ds  = tokenize(tokenizer, test_df)

    args = TrainingArguments(
        output_dir              = OUTPUT_DIR,
        num_train_epochs        = EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        learning_rate           = LR,
        weight_decay            = 0.01,
        evaluation_strategy     = "epoch",
        save_strategy           = "epoch",
        load_best_model_at_end  = True,
        metric_for_best_model   = "f1",
        logging_dir             = os.path.join(OUTPUT_DIR, "logs"),
        logging_steps           = 50,
        seed                    = SEED,
        fp16                    = (device == "cuda"),
    )

    trainer = Trainer(
        model           = model,
        args            = args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        compute_metrics = compute_metrics,
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("\n🚀 Starting BERT fine-tuning...")
    trainer.train()

    # Final test evaluation
    preds_out = trainer.predict(test_ds)
    preds     = np.argmax(preds_out.predictions, axis=-1)
    print("\n📈 Test Set Results:")
    print(classification_report(test_df["label"].values, preds, target_names=["FAKE", "TRUE"]))

    # Save model + tokenizer
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n✅ Model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
