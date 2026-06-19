"""
================================================================================
 MOVIE REVIEW SENTIMENT ANALYSIS SYSTEM
 FILE: flask_app.py
 ROLE: Production Flask backend.

 RESPONSIBILITIES
   1. Load the four pre-trained artefacts (TF-IDF + Naive Bayes + Logistic
      Regression + Linear SVM) once, at start-up.
   2. Run the full inference pipeline for an incoming review:
          raw text -> clean -> vectorise -> 4 models -> ensemble vote
   3. Serve the single-page cinematic frontend (templates/index.html).
   4. Expose a JSON prediction API at  POST /predict
      and a lightweight health probe at  GET /health.

 DESIGN NOTES
   - All file paths are built from THIS file's location (BASE_DIR), so the app
     runs correctly no matter what the current working directory is. This is
     the main reason it previously failed when launched from Jupyter.
   - The preprocessing here MUST stay byte-for-byte identical to the cleaning
     used in train_models.py, otherwise the vocabulary will not line up with
     the trained model weights and predictions become meaningless.
================================================================================
"""

from __future__ import annotations

import os
import re
import time
import logging

import numpy as np
import joblib

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

from textblob import TextBlob

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag


# ──────────────────────────────────────────────────────────────────────────────
#  PATHS  (absolute, derived from this file — fixes "works in terminal, not in
#          Jupyter" path bugs)
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")


# ──────────────────────────────────────────────────────────────────────────────
#  LOGGING
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sentiment")


# ──────────────────────────────────────────────────────────────────────────────
#  NLTK RESOURCES  (downloaded quietly, only the first time)
# ──────────────────────────────────────────────────────────────────────────────
def ensure_nltk_resources() -> None:
    """Download the NLP reference data the preprocessing pipeline depends on."""
    packages = [
        "stopwords",
        "punkt",
        "punkt_tab",
        "wordnet",
        "omw-1.4",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
    ]
    for pkg in packages:
        nltk.download(pkg, quiet=True)


ensure_nltk_resources()


# ──────────────────────────────────────────────────────────────────────────────
#  MODEL REGISTRY  (loaded once at import time)
# ──────────────────────────────────────────────────────────────────────────────
def load_artifacts() -> dict:
    """Load every trained artefact into memory and return them in a dict."""
    required = {
        "tfidf": "tfidf_vectorizer.joblib",
        "nb": "naive_bayes.joblib",
        "lr": "logistic_regression.joblib",
        "svm": "svm_linearsvc.joblib",
    }

    artifacts: dict = {}
    log.info("Loading trained artefacts from %s", MODELS_DIR)
    for key, filename in required.items():
        path = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing model file: {path}\n"
                f"Run train_models.py first to regenerate the .joblib files."
            )
        artifacts[key] = joblib.load(path)
        log.info("  loaded %-22s (%s)", key, filename)

    log.info("All 4 artefacts loaded successfully.")
    return artifacts


ARTIFACTS = load_artifacts()
TFIDF = ARTIFACTS["tfidf"]
NB_MODEL = ARTIFACTS["nb"]
LR_MODEL = ARTIFACTS["lr"]
SVM_MODEL = ARTIFACTS["svm"]


# ──────────────────────────────────────────────────────────────────────────────
#  PREPROCESSING  (must match train_models.py EXACTLY)
# ──────────────────────────────────────────────────────────────────────────────
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

_WORDNET_MAP = {"J": wordnet.ADJ, "V": wordnet.VERB, "N": wordnet.NOUN, "R": wordnet.ADV}


def get_wordnet_pos(treebank_tag: str):
    """Map an NLTK Treebank POS tag to the WordNet tag the lemmatizer expects."""
    return _WORDNET_MAP.get(treebank_tag[0], wordnet.NOUN)


def preprocess_text(text: str) -> str:
    """
    Clean a single review the same way the training corpus was cleaned:
    lowercase -> strip HTML -> keep letters only -> tokenize -> POS tag ->
    drop stopwords / short tokens -> lemmatize -> rejoin.
    """
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)

    cleaned = [
        LEMMATIZER.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in tagged
        if word not in STOP_WORDS and len(word) > 2
    ]
    return " ".join(cleaned)


# ──────────────────────────────────────────────────────────────────────────────
#  INFERENCE PIPELINE
# ──────────────────────────────────────────────────────────────────────────────
def _label(value: int) -> str:
    return "POSITIVE" if value == 1 else "NEGATIVE"


def analyse_review(review: str) -> dict:
    """
    Run the entire ensemble pipeline on one raw review and return a structured
    result dictionary ready to be JSON-serialised for the frontend.
    """
    started = time.perf_counter()

    # 1) TextBlob — rule based, reads the RAW text.
    blob = TextBlob(review).sentiment
    tb_polarity = float(blob.polarity)
    tb_subjectivity = float(blob.subjectivity)
    tb_pred = 1 if tb_polarity >= 0 else 0
    # Confidence scales with the strength of the polarity signal.
    tb_confidence = round(min(abs(tb_polarity) * 100 + 50, 99.9), 1)

    # 2) ML models — operate on the cleaned + vectorised text.
    cleaned = preprocess_text(review)
    vector = TFIDF.transform([cleaned])

    nb_pred = int(NB_MODEL.predict(vector)[0])
    nb_confidence = round(float(np.max(NB_MODEL.predict_proba(vector)[0])) * 100, 1)

    lr_pred = int(LR_MODEL.predict(vector)[0])
    lr_confidence = round(float(np.max(LR_MODEL.predict_proba(vector)[0])) * 100, 1)

    # LinearSVC has no predict_proba — convert the signed distance from the
    # decision boundary into a 0-100 confidence using a sigmoid.
    svm_pred = int(SVM_MODEL.predict(vector)[0])
    svm_distance = float(SVM_MODEL.decision_function(vector)[0])
    svm_confidence = round(float(1 / (1 + np.exp(-abs(svm_distance)))) * 100, 1)

    # 3) Majority vote across all 4 approaches.
    votes = [tb_pred, nb_pred, lr_pred, svm_pred]
    positive_votes = sum(votes)
    final_pred = 1 if positive_votes >= 2 else 0
    agree_count = positive_votes if final_pred == 1 else (4 - positive_votes)

    # 4) Aggregate ensemble confidence (mean of the models agreeing).
    per_model = {
        "textblob": (tb_pred, tb_confidence),
        "naive_bayes": (nb_pred, nb_confidence),
        "logistic_regression": (lr_pred, lr_confidence),
        "svm": (svm_pred, svm_confidence),
    }
    agreeing = [c for (p, c) in per_model.values() if p == final_pred]
    ensemble_confidence = round(sum(agreeing) / len(agreeing), 1) if agreeing else 0.0

    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)

    return {
        "final_prediction": _label(final_pred),
        "ensemble_confidence": ensemble_confidence,
        "agree_count": agree_count,
        "total_models": 4,
        "polarity_score": round(tb_polarity, 3),
        "subjectivity": round(tb_subjectivity, 3),
        "word_count": len(review.split()),
        "char_count": len(review),
        "processing_time_ms": elapsed_ms,
        "cleaned_text": cleaned[:300] + ("..." if len(cleaned) > 300 else ""),
        "models": {
            "textblob": {"prediction": _label(tb_pred), "confidence": tb_confidence},
            "naive_bayes": {"prediction": _label(nb_pred), "confidence": nb_confidence},
            "logistic_regression": {"prediction": _label(lr_pred), "confidence": lr_confidence},
            "svm": {"prediction": _label(svm_pred), "confidence": svm_confidence},
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
#  FLASK APP
# ──────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    """Serve the single-page cinematic frontend."""
    return render_template("index.html")


@app.route("/health")
def health():
    """Lightweight readiness probe — useful for demos and monitoring."""
    return jsonify({"status": "ok", "models_loaded": 4})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept a JSON body  { "review": "<text>" }  and return the full ensemble
    analysis as JSON. Returns 400 for invalid input, 500 for unexpected errors.
    """
    try:
        payload = request.get_json(silent=True) or {}
        review = str(payload.get("review", "")).strip()

        if len(review) < 5:
            return jsonify({"error": "Please enter a movie review of at least 5 characters."}), 400

        result = analyse_review(review)
        return jsonify(result)

    except Exception as exc:  # noqa: BLE001 — surface a clean message to the client
        log.exception("Prediction failed")
        return jsonify({"error": f"Server error: {exc}"}), 500


# ──────────────────────────────────────────────────────────────────────────────
#  ENTRY POINTS
# ──────────────────────────────────────────────────────────────────────────────
def run_app(host: str = "127.0.0.1", port: int = 5000, debug: bool = False) -> None:
    """
    Start the server.

    IMPORTANT (Jupyter): the Werkzeug auto-reloader (enabled by debug=True)
    restarts the process via sys.argv, which does not exist inside a notebook
    kernel — that is why the server appeared to "run" but never printed a URL.
    We therefore force use_reloader=False so it works identically from a
    terminal OR a Jupyter cell.
    """
    banner = (
        "\n" + "=" * 62 + "\n"
        "   MOVIE SENTIMENT ANALYSIS  —  server starting\n"
        f"   Open your browser at:  http://{host}:{port}\n"
        + "=" * 62 + "\n"
    )
    print(banner)
    app.run(host=host, port=port, debug=debug, use_reloader=False)


if __name__ == "__main__":
    # Running from a terminal:  python flask_app.py
    run_app(host="127.0.0.1", port=5000, debug=True)
