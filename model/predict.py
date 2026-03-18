# ============================================================
#  predict.py — Ensemble Prediction Engine
#  Models: BERT (primary) → TF-IDF + NB + RF (fallback)
#  Author : Salar Ahmed | FYP 2025-2026
# ============================================================

import os, pickle, datetime
import numpy as np
from utils.preprocess import clean_urdu_text

# ── Paths ────────────────────────────────────────────────────
MODEL_DIR   = os.path.join(os.path.dirname(__file__))
TFIDF_PATH  = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
NB_PATH     = os.path.join(MODEL_DIR, "nb_model.pkl")
RF_PATH     = os.path.join(MODEL_DIR, "rf_model.pkl")
BERT_PATH   = os.path.join(MODEL_DIR, "bert_model")

# ── Lazy-load models ─────────────────────────────────────────
_tfidf = _nb = _rf = _bert_pipeline = None

def _load_sklearn_models():
    global _tfidf, _nb, _rf
    if _tfidf is None:
        if os.path.exists(TFIDF_PATH) and os.path.exists(NB_PATH) and os.path.exists(RF_PATH):
            with open(TFIDF_PATH, "rb") as f: _tfidf = pickle.load(f)
            with open(NB_PATH,   "rb") as f: _nb    = pickle.load(f)
            with open(RF_PATH,   "rb") as f: _rf    = pickle.load(f)
        else:
            _train_and_save()


def _train_and_save():
    """Auto-train TF-IDF + NB + RF from CSV datasets if models not present."""
    global _tfidf, _nb, _rf
    import pandas as pd
    from sklearn.pipeline import make_pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    DATA_DIR = r"C:\Users\KABIR BALOCH\Downloads\fake_news_complete_project\fake_news_project"
    true_path = os.path.join(DATA_DIR, "True.csv")
    fake_path = os.path.join(DATA_DIR, "Fake.csv")

    if not os.path.exists(true_path) or not os.path.exists(fake_path):
        print("⚠️  CSVs not found — using dummy fallback model.")
        _tfidf = _nb = _rf = None
        return

    true_df = pd.read_csv(true_path); true_df["label"] = 1   # 1 = True
    fake_df = pd.read_csv(fake_path); fake_df["label"] = 0   # 0 = Fake
    df = pd.concat([true_df, fake_df], ignore_index=True)

    text_col = "text" if "text" in df.columns else df.columns[0]
    df["clean"] = df[text_col].fillna("").apply(clean_urdu_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label"], test_size=0.2, random_state=42
    )

    # TF-IDF vectoriser shared by both classifiers
    from sklearn.feature_extraction.text import TfidfVectorizer
    _tfidf = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2))
    X_train_vec = _tfidf.fit_transform(X_train)
    X_test_vec  = _tfidf.transform(X_test)

    # Naive Bayes
    _nb = MultinomialNB(); _nb.fit(X_train_vec, y_train)

    # Random Forest
    _rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    _rf.fit(X_train_vec, y_train)

    nb_acc = _nb.score(X_test_vec, y_test)
    rf_acc = _rf.score(X_test_vec, y_test)
    print(f"✅ Naive Bayes accuracy : {nb_acc:.3f}")
    print(f"✅ Random Forest accuracy: {rf_acc:.3f}")

    with open(TFIDF_PATH, "wb") as f: pickle.dump(_tfidf, f)
    with open(NB_PATH,    "wb") as f: pickle.dump(_nb,    f)
    with open(RF_PATH,    "wb") as f: pickle.dump(_rf,    f)
    print("✅ Models saved to disk.")


def _load_bert():
    """Load fine-tuned BERT model if available."""
    global _bert_pipeline
    if _bert_pipeline is not None:
        return True
    if not os.path.exists(BERT_PATH):
        return False
    try:
        from transformers import pipeline
        _bert_pipeline = pipeline(
            "text-classification",
            model=BERT_PATH,
            tokenizer=BERT_PATH,
            truncation=True,
            max_length=512,
        )
        print("✅ BERT model loaded.")
        return True
    except Exception as e:
        print(f"⚠️  BERT load failed: {e}")
        return False


# ── Public API ───────────────────────────────────────────────

def predict_news(text: str) -> dict:
    """
    Returns:
        {
          label       : "FAKE" | "TRUE",
          confidence  : float (0-1),
          bert_vote   : "FAKE" | "TRUE" | "N/A",
          nb_vote     : "FAKE" | "TRUE",
          rf_vote     : "FAKE" | "TRUE",
          model_used  : str,
          timestamp   : str (ISO),
        }
    """
    _load_sklearn_models()
    cleaned = clean_urdu_text(text)

    votes = {}
    confidences = []

    # ── BERT vote ────────────────────────────────────────────
    if _load_bert():
        try:
            bert_out = _bert_pipeline(cleaned[:512])[0]
            bert_label = bert_out["label"].upper()
            bert_conf  = bert_out["score"]
            # Map model labels to FAKE/TRUE (adjust for your fine-tuned labels)
            if bert_label in ("LABEL_1", "TRUE", "REAL"):
                votes["bert"] = 1; bert_display = "TRUE"
            else:
                votes["bert"] = 0; bert_display = "FAKE"
            confidences.append(bert_conf)
        except Exception:
            bert_display = "N/A"
    else:
        bert_display = "N/A"

    # ── TF-IDF + NB + RF votes ───────────────────────────────
    if _tfidf is not None:
        vec = _tfidf.transform([cleaned])

        nb_pred  = int(_nb.predict(vec)[0])
        nb_prob  = float(_nb.predict_proba(vec)[0][nb_pred])
        votes["nb"] = nb_pred
        confidences.append(nb_prob)

        rf_pred  = int(_rf.predict(vec)[0])
        rf_prob  = float(_rf.predict_proba(vec)[0][rf_pred])
        votes["rf"] = rf_pred
        confidences.append(rf_prob)

        nb_display = "TRUE" if nb_pred == 1 else "FAKE"
        rf_display = "TRUE" if rf_pred == 1 else "FAKE"
    else:
        # Dummy fallback when no models or CSVs are available
        nb_display = rf_display = "N/A"
        votes["nb"] = votes["rf"] = 0

    # ── Ensemble: majority vote ──────────────────────────────
    if votes:
        final_label_int = int(np.round(np.mean(list(votes.values()))))
    else:
        final_label_int = 0

    final_label = "TRUE" if final_label_int == 1 else "FAKE"
    avg_confidence = float(np.mean(confidences)) if confidences else 0.5

    model_used = "BERT+NB+RF" if bert_display != "N/A" else ("NB+RF" if _tfidf else "dummy")

    return {
        "label"      : final_label,
        "confidence" : round(avg_confidence, 4),
        "bert_vote"  : bert_display,
        "nb_vote"    : nb_display,
        "rf_vote"    : rf_display,
        "model_used" : model_used,
        "timestamp"  : datetime.datetime.utcnow().isoformat(),
    }
