"""
train_model.py
--------------
Trains the spam / ham classifier.

Pipeline:
    spam.csv  ->  column detection  ->  cleaning  ->  tokenization
              ->  stopword removal  ->  TF-IDF  ->  MultinomialNB
              ->  evaluation  ->  saved .pkl files

Run from the project folder:
    python train_model.py

app.py imports clean_text() from this file, so training and prediction
always use exactly the same preprocessing.
"""

import os
import re
import sys

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# ----------------------------------------------------------------------
# Paths (built with os.path.join so they work on Windows and Linux)
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "spam.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "spam_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

RANDOM_STATE = 42
TEST_SIZE = 0.2

# ----------------------------------------------------------------------
# NLTK setup
# ----------------------------------------------------------------------
# The project must not crash just because NLTK data is missing, so we try
# to download it once and fall back to a built-in stopword list and a
# regex tokenizer if there is no internet connection.
FALLBACK_STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "s", "t", "can", "will", "just", "don", "should", "now",
}


def _setup_nltk():
    """Return (tokenizer_function, stopword_set)."""
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize

        for resource, path in [
            ("stopwords", "corpora/stopwords"),
            ("punkt", "tokenizers/punkt"),
            ("punkt_tab", "tokenizers/punkt_tab"),
        ]:
            try:
                nltk.data.find(path)
            except LookupError:
                try:
                    nltk.download(resource, quiet=True)
                except Exception:
                    pass

        stop_set = set(stopwords.words("english"))
        # Make sure the tokenizer really works before we commit to it.
        word_tokenize("this is a test")
        return word_tokenize, stop_set
    except Exception:
        print("[warn] NLTK data unavailable - using built-in tokenizer/stopwords.")
        return (lambda text: re.findall(r"[a-z]+", text)), FALLBACK_STOPWORDS


TOKENIZER, STOPWORDS = _setup_nltk()

# ----------------------------------------------------------------------
# Text cleaning  (used by BOTH training and prediction)
# ----------------------------------------------------------------------
URL_PATTERN = re.compile(r"(http\S+|https\S+|www\.\S+)")
EMAIL_PATTERN = re.compile(r"\S+@\S+")
NON_ALPHA_PATTERN = re.compile(r"[^a-z\s]")


def clean_text(text):
    """
    lowercase -> remove URLs/emails -> remove punctuation, digits and
    special characters -> tokenize -> remove stopwords and 1-letter tokens.

    Always returns a string (possibly empty), never None.
    """
    if text is None:
        return ""
    text = str(text).lower()
    text = URL_PATTERN.sub(" ", text)
    text = EMAIL_PATTERN.sub(" ", text)
    text = NON_ALPHA_PATTERN.sub(" ", text)

    try:
        tokens = TOKENIZER(text)
    except Exception:
        tokens = text.split()

    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)


# ----------------------------------------------------------------------
# Dataset loading
# ----------------------------------------------------------------------
TEXT_COLUMN_CANDIDATES = [
    "text", "message", "email", "email_text", "body", "content",
    "sms", "msg", "v2",
]
LABEL_COLUMN_CANDIDATES = [
    "label", "category", "class", "target", "spam", "type", "v1",
]

SPAM_WORDS = {"spam", "1", "1.0", "true", "yes"}
HAM_WORDS = {"ham", "0", "0.0", "false", "no", "not spam", "notspam", "legit"}


def load_dataset(path):
    """Read the CSV with several encodings so odd files still open."""
    if not os.path.exists(path):
        sys.exit(
            f"[error] Dataset not found at: {path}\n"
            "Put your spam.csv inside the 'dataset' folder and run again."
        )

    last_error = None
    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(path, encoding=encoding)
            print(f"[info] Loaded dataset using encoding '{encoding}'.")
            return df
        except UnicodeDecodeError as err:
            last_error = err
        except Exception as err:
            last_error = err
            break
    sys.exit(f"[error] Could not read the CSV file: {last_error}")


def detect_columns(df):
    """
    Find the text column and the label column.
    Never silently guesses: it prints what it chose, and stops if unsure.
    """
    lower_map = {str(c).strip().lower(): c for c in df.columns}

    text_col = next((lower_map[c] for c in TEXT_COLUMN_CANDIDATES if c in lower_map), None)
    label_col = next((lower_map[c] for c in LABEL_COLUMN_CANDIDATES if c in lower_map), None)

    # "spam" can be either a 0/1 label column or a text column name, so if it
    # was picked for both roles, keep it as the label.
    if text_col is not None and text_col == label_col:
        text_col = None

    if text_col is None or label_col is None:
        # Last resort: the label column is the one with exactly 2 unique values
        # and the text column is the one with the longest average strings.
        for col in df.columns:
            if label_col is None and df[col].nunique(dropna=True) == 2:
                label_col = col
        if text_col is None:
            lengths = {
                col: df[col].astype(str).str.len().mean()
                for col in df.columns
                if col != label_col
            }
            if lengths:
                text_col = max(lengths, key=lengths.get)

    if text_col is None or label_col is None:
        sys.exit(
            "[error] Could not identify the text and label columns.\n"
            f"Columns found: {list(df.columns)}\n"
            "Rename your columns to 'message' and 'label' and run again."
        )

    print(f"[info] Text column  : '{text_col}'")
    print(f"[info] Label column : '{label_col}'")
    print(f"[info] Sample labels: {df[label_col].dropna().unique()[:5]}")
    return text_col, label_col


def normalise_labels(series):
    """spam/ham or 1/0 (in any case, with stray spaces) -> 1 / 0."""
    def to_binary(value):
        v = str(value).strip().lower()
        if v in SPAM_WORDS:
            return 1
        if v in HAM_WORDS:
            return 0
        return None

    return series.map(to_binary)


def prepare_dataframe(df, text_col, label_col):
    data = df[[text_col, label_col]].copy()
    data.columns = ["text", "label"]

    before = len(data)
    data["label"] = normalise_labels(data["label"])
    data = data.dropna(subset=["text", "label"])
    data = data.drop_duplicates()
    data["label"] = data["label"].astype(int)

    data["clean_text"] = data["text"].apply(clean_text)
    data = data[data["clean_text"].str.strip() != ""]

    print(f"[info] Rows: {before} -> {len(data)} after cleaning, "
          "removing missing values and duplicates.")
    print(f"[info] Ham (0): {(data['label'] == 0).sum()} | "
          f"Spam (1): {(data['label'] == 1).sum()}")

    if len(data) < 20 or data["label"].nunique() < 2:
        sys.exit("[error] Not enough usable data (need both classes and 20+ rows).")
    return data


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
def main():
    print("=" * 60)
    print("SPAM EMAIL CLASSIFIER - TRAINING")
    print("=" * 60)

    df = load_dataset(DATASET_PATH)
    print(f"[info] Shape: {df.shape}")
    print(f"[info] Columns: {list(df.columns)}")

    text_col, label_col = detect_columns(df)
    data = prepare_dataframe(df, text_col, label_col)

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        data["clean_text"],
        data["label"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=data["label"],
    )

    # TF-IDF turns each cleaned email into a row of numbers. A word gets a
    # high score when it appears often in THIS email but rarely across all
    # the other emails, so words like "free" or "winner" become strong
    # signals while common words stay unimportant.
    # min_df=2 drops words that appear in only one email (usually typos and
    # random strings), which keeps the feature space clean.
    vectorizer = TfidfVectorizer(ngram_range=(1, 1), min_df=2, sublinear_tf=True)
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    print(f"[info] TF-IDF features: {len(vectorizer.get_feature_names_out())}")

    # alpha is Laplace smoothing: it stops a single unseen word from making a
    # probability zero. fit_prior=False tells the model not to assume spam is
    # rare just because the dataset has more ham than spam, which raises recall
    # (fewer spam emails slip through as ham).
    model = MultinomialNB(alpha=0.1, fit_prior=False)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n" + "-" * 60)
    print("EVALUATION ON THE TEST SET (real numbers from this run)")
    print("-" * 60)
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1-score : {f1_score(y_test, y_pred, zero_division=0):.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix")
    print("                 Predicted HAM   Predicted SPAM")
    print(f"Actual HAM   {cm[0][0]:>12}   {cm[0][1]:>14}")
    print(f"Actual SPAM  {cm[1][0]:>12}   {cm[1][1]:>14}")

    print("\nClassification report")
    print(classification_report(y_test, y_pred, target_names=["HAM", "SPAM"],
                                zero_division=0))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"[saved] {MODEL_PATH}")
    print(f"[saved] {VECTORIZER_PATH}")
    print("\nTraining finished. Now run:  python app.py")


if __name__ == "__main__":
    main()
