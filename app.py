"""
app.py
------
Flask web application for the spam classifier.

Routes:
    GET  /         -> the web page
    POST /predict  -> JSON in, JSON out (used by static/script.js)

Run from the project folder (after training the model):
    python app.py
Then open http://127.0.0.1:5000
"""

import os
import re

import joblib
from flask import Flask, jsonify, render_template, request

# The SAME cleaning function that was used during training.
from train_model import clean_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "spam_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "model", "tfidf_vectorizer.pkl")

MAX_INPUT_CHARS = 20000

app = Flask(__name__)

# ----------------------------------------------------------------------
# Load the trained files once, when the server starts
# ----------------------------------------------------------------------
model = None
vectorizer = None
load_error = None

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("[info] Model and vectorizer loaded.")
except FileNotFoundError:
    load_error = (
        "Model files were not found. Run 'python train_model.py' first, "
        "then start the app again."
    )
    print("[error]", load_error)
except Exception as err:  # corrupted file, version mismatch, etc.
    load_error = f"Model files could not be loaded ({err}). Please retrain."
    print("[error]", load_error)


# ----------------------------------------------------------------------
# Explainability layer
# ----------------------------------------------------------------------
# IMPORTANT: these rules are NOT the internals of the Naive Bayes model.
# The model makes the SPAM/HAM decision on its own. This is a separate,
# human-readable layer that looks at the raw email for patterns people
# usually associate with spam, so the result is easier to understand.
URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)

RULES = [
    ("Prize or reward language detected",
     r"\b(winner|won|win|prize|lottery|jackpot|reward|congratulations|congrats|giveaway)\b"),
    ("Promotional or marketing language detected",
     r"\b(free|offer|discount|sale|deal|cheap|buy|order now|limited time|exclusive|coupon|% off)\b"),
    ("Urgent call-to-action detected",
     r"\b(urgent|immediately|act now|hurry|last chance|final notice|expires|don'?t miss|apply now|click here|claim now|call now)\b"),
    ("Money related terms detected",
     r"(\$|₹|\b(cash|usd|dollars|rupees|income|profit|earn|loan|refund|payment|bitcoin|investment)\b)"),
    ("Account, password or security request detected",
     r"\b(password|verify|verification|account (suspended|blocked|closed)|login|otp|cvv|bank details|update your (details|payment)|credit card)\b"),
]


def build_explanation(raw_text):
    """Return a list of short human-readable reasons."""
    reasons = []
    lowered = raw_text.lower()

    for message, pattern in RULES:
        if re.search(pattern, lowered, re.IGNORECASE):
            reasons.append(message)

    links = URL_RE.findall(raw_text)
    if links:
        reasons.append(
            f"{len(links)} suspicious link{'s' if len(links) > 1 else ''} found in the text"
        )

    letters = [c for c in raw_text if c.isalpha()]
    if len(letters) >= 20:
        caps_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
        if caps_ratio > 0.3:
            reasons.append("Excessive use of capital letters")

    if raw_text.count("!") >= 3:
        reasons.append("Excessive exclamation marks")

    return reasons


def build_stats(raw_text):
    return {
        "words": len(raw_text.split()),
        "characters": len(raw_text),
        "links": len(URL_RE.findall(raw_text)),
        "exclamations": raw_text.count("!"),
    }


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", model_ready=(model is not None))


@app.route("/predict", methods=["POST"])
def predict():
    if model is None or vectorizer is None:
        return jsonify({"success": False, "error": load_error}), 503

    # Accept both fetch/JSON and a normal HTML form post.
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        email_text = payload.get("email_text", "")
    else:
        email_text = request.form.get("email_text", "")

    if not isinstance(email_text, str):
        return jsonify({"success": False,
                        "error": "Invalid input. Please send plain text."}), 400

    email_text = email_text.strip()

    if not email_text:
        return jsonify({"success": False,
                        "error": "Please paste some email text before analysing."}), 400

    if len(email_text) > MAX_INPUT_CHARS:
        return jsonify({"success": False,
                        "error": f"Email is too long. Limit is {MAX_INPUT_CHARS} characters."}), 400

    try:
        cleaned = clean_text(email_text)
        if not cleaned.strip():
            # Everything was punctuation, digits or stopwords.
            return jsonify({
                "success": False,
                "error": "No usable words found after cleaning. Please enter real email text."
            }), 400

        features = vectorizer.transform([cleaned])
        prediction = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]
        confidence = round(float(probabilities[prediction]) * 100, 1)

        result = {
            "success": True,
            "label": "SPAM" if prediction == 1 else "HAM",
            "is_spam": prediction == 1,
            "confidence": confidence,
            "spam_probability": round(float(probabilities[1]) * 100, 1),
            "stats": build_stats(email_text),
            "reasons": build_explanation(email_text) if prediction == 1 else [],
        }
        return jsonify(result)

    except Exception as err:
        print("[error] Prediction failed:", err)
        return jsonify({
            "success": False,
            "error": "Something went wrong while analysing this email. Please try again."
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
