# ============================================================
#  preprocess.py — Urdu Text Cleaning & Normalization
#  Author : Salar Ahmed | FYP 2025-2026
# ============================================================

import re, unicodedata

# ── Urdu-specific character sets ─────────────────────────────
URDU_DIACRITICS = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]"
)
ARABIC_INDIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
EXTENDED_ARABIC     = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")

# ── Common Urdu character normalisation map ──────────────────
NORMALISE_MAP = str.maketrans({
    "\u0622": "\u0627",   # آ → ا  (Alef Madda → Alef)
    "\u0623": "\u0627",   # أ → ا
    "\u0625": "\u0627",   # إ → ا
    "\u0629": "\u0647",   # ة → ه  (Ta Marbuta → He)
    "\u0649": "\u06CC",   # ى → ی  (Alef Maqsura → Ye)
    "\u0643": "\u06A9",   # ك → ک  (Arabic Kaf → Urdu Kaf)
    "\u06C0": "\u06BE",   # ۀ → ھ  (He with Ye above → Docharshmi He)
})

# ── Stopwords (common Urdu function words) ───────────────────
URDU_STOPWORDS = {
    "اور", "کہ", "ہے", "کی", "میں", "کو", "نے", "سے", "پر", "ہیں",
    "تھا", "تھی", "تھے", "ہوں", "ہو", "گا", "گی", "گے", "جو", "یہ",
    "وہ", "اس", "ان", "کے", "کر", "بھی", "تو", "لیے", "ایک", "لیکن",
    "مگر", "بھی", "نہیں", "ہر", "ہاں", "نہ", "ہی", "اب", "پھر", "اگر",
    "the", "a", "an", "is", "are", "was", "were", "in", "of", "to",
}


def remove_diacritics(text: str) -> str:
    """Remove Urdu/Arabic diacritics (harakat)."""
    return URDU_DIACRITICS.sub("", text)


def normalise_chars(text: str) -> str:
    """Normalise Unicode character variants common in Urdu text."""
    return text.translate(NORMALISE_MAP)


def normalise_digits(text: str) -> str:
    """Convert Arabic-Indic and Extended Arabic digits to ASCII."""
    return text.translate(ARABIC_INDIC_DIGITS).translate(EXTENDED_ARABIC)


def remove_urls(text: str) -> str:
    return re.sub(r"https?://\S+|www\.\S+", " ", text)


def remove_emails(text: str) -> str:
    return re.sub(r"\S+@\S+\.\S+", " ", text)


def remove_hashtags_mentions(text: str) -> str:
    return re.sub(r"[@#]\S+", " ", text)


def remove_punctuation(text: str) -> str:
    """Keep Urdu sentence-ending marks (۔ ؟) but remove others."""
    return re.sub(r"[^\w\s\u0600-\u06FF۔؟]", " ", text)


def remove_extra_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def remove_stopwords(text: str) -> str:
    tokens = text.split()
    return " ".join(t for t in tokens if t not in URDU_STOPWORDS)


def clean_urdu_text(
    text: str,
    *,
    diacritics: bool = True,
    normalise: bool  = True,
    digits: bool     = True,
    urls: bool       = True,
    emails: bool     = True,
    social: bool     = True,
    punctuation: bool = True,
    stopwords: bool  = False,   # Off by default for BERT
) -> str:
    """
    Full preprocessing pipeline for Urdu/English news text.

    Parameters
    ----------
    text        : Raw news string (Urdu, English, or mixed).
    diacritics  : Remove Urdu harakat.
    normalise   : Normalise Unicode character variants.
    digits      : Transliterate Arabic-Indic numerals to ASCII.
    urls        : Strip URLs.
    emails      : Strip email addresses.
    social      : Strip hashtags and @mentions.
    punctuation : Remove non-Urdu punctuation.
    stopwords   : Remove common Urdu stopwords (use for TF-IDF, not BERT).

    Returns
    -------
    str : Cleaned text, NFC-normalised.
    """
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFC", text)

    if diacritics:  text = remove_diacritics(text)
    if normalise:   text = normalise_chars(text)
    if digits:      text = normalise_digits(text)
    if urls:        text = remove_urls(text)
    if emails:      text = remove_emails(text)
    if social:      text = remove_hashtags_mentions(text)
    if punctuation: text = remove_punctuation(text)
    if stopwords:   text = remove_stopwords(text)

    return remove_extra_whitespace(text)


# ── Quick test ───────────────────────────────────────────────
if __name__ == "__main__":
    sample = "عمران خان نے کہا کہ https://geo.tv پر خبر جھوٹی ہے! #FakeNews @ARYNews"
    print("Original:", sample)
    print("Cleaned :", clean_urdu_text(sample))
    print("No stops:", clean_urdu_text(sample, stopwords=True))
