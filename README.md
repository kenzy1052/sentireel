# 🎬 Movie Review Sentiment Analysis

> A premium, cinematic AI web application that reads a movie review and predicts
> whether it is **positive** or **negative** — fusing four independent NLP / ML
> models into one confident verdict.

Built with **Python · Flask · NLTK · scikit-learn · TextBlob** and a custom,
Netflix-style dark UI (HTML / CSS / JavaScript).

---

## ✨ Features

- **Ensemble intelligence** — TextBlob, Naive Bayes, Logistic Regression and a
  Linear SVM each vote; a majority vote produces the final verdict.
- **Cinematic dark UI** — glassmorphism input card, animated loading sequence,
  confidence ring and animated per-model progress bars.
- **Live insight** — words analysed, character count, processing time and
  TextBlob polarity for every prediction.
- **Robust backend** — absolute model paths, health check, input validation and
  clean JSON error handling.
- **Runs anywhere** — terminal **or** Jupyter Notebook (the classic Jupyter
  "no URL appears" bug is fixed — see below).

---

## 📁 Project structure

```
movie_sentiment/
│
├── flask_app.py              # Flask backend: loads models, serves UI, /predict API
├── train_models.py          # Re-train the models from the IMDB dataset (optional)
│
├── models/                  # Pre-trained artefacts (ship with the project)
│   ├── tfidf_vectorizer.joblib
│   ├── naive_bayes.joblib
│   ├── logistic_regression.joblib
│   ├── svm_linearsvc.joblib
│   └── count_vectorizer.joblib
│
├── templates/
│   └── index.html           # Single-page cinematic frontend
│
├── static/
│   ├── style.css            # Dark cinematic theme
│   ├── script.js            # Counters, loading sequence, animated results
│   └── images/
│       └── hero-cinema.jpg  # Hero background
│
├── notebooks/
│   ├── 01_model_training_and_eda.ipynb   # Original training + EDA notebook
│   └── Run_Flask_App.ipynb               # One-cell launcher for Jupyter
│
├── results/                 # Saved EDA / evaluation charts
├── screenshots/             # App screenshots for the report
├── dataset/                 # ⬅ put "IMDB Dataset.csv" here (not included)
│
├── requirements.txt
└── README.md
```

---

## 🚀 Quick start

### 1. (Recommended) Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> The first run automatically downloads the small NLTK data packages
> (stopwords, punkt, wordnet, POS tagger). This needs an internet connection
> once.

### 3. Run the app

**Option A — from a terminal:**

```bash
python flask_app.py
```

**Option B — from Jupyter Notebook:** open
`notebooks/Run_Flask_App.ipynb` and run the single cell.

### 4. Open the application

Go to **http://127.0.0.1:5000** in your browser, paste a review and click
**Analyse Review**.

> The web app does **not** need the IMDB dataset — it uses the pre-trained
> models in `models/`.

---

## 🪐 Running from Jupyter Notebook (and why it failed before)

**The problem:** the original code called `app.run(debug=True)`. With
`debug=True`, Flask/Werkzeug enables an **auto-reloader** that restarts the
server by re-launching the process via `sys.argv`. Inside a Jupyter kernel
there is no script process to relaunch, so the reloader silently fails — the
cell shows "running" forever and the `http://127.0.0.1:5000` URL never appears.

**The fix:** `flask_app.py` now exposes a `run_app()` helper that calls
`app.run(..., use_reloader=False)`. This behaves identically in a terminal and
in a notebook.

Minimal notebook cell (already provided in `notebooks/Run_Flask_App.ipynb`):

```python
import os
os.chdir(r"PATH/TO/movie_sentiment")   # folder that contains flask_app.py

from flask_app import run_app
run_app(host="127.0.0.1", port=5000)   # then open http://127.0.0.1:5000
```

To stop the server, use **Kernel → Interrupt** (or the ■ stop button).

---

## 🧠 How the prediction pipeline works

```
raw review
   ├─► TextBlob ............ reads emotional polarity from the raw text
   └─► clean & lemmatise ──► TF-IDF (10,000 features)
                               ├─► Naive Bayes        → vote + confidence
                               ├─► Logistic Regression→ vote + confidence
                               └─► Linear SVM          → vote + confidence
                                         │
                                  majority vote → FINAL VERDICT + confidence
```

The text cleaning in `flask_app.py` is **identical** to `train_models.py`
(lowercase → strip HTML → letters only → tokenize → POS-tag → remove
stopwords → lemmatize), so the vocabulary always lines up with the trained
model weights.

---

## 🔁 Re-training the models (optional)

1. Put `IMDB Dataset.csv` in `dataset/` (see `dataset/README.md`).
2. Run:

   ```bash
   python train_models.py
   ```

3. New `.joblib` files are written to `models/`.

---

## 🔌 API reference

### `POST /predict`

Request body:

```json
{ "review": "An absolute masterpiece, beautifully acted." }
```

Response:

```json
{
  "final_prediction": "POSITIVE",
  "ensemble_confidence": 88.4,
  "agree_count": 4,
  "total_models": 4,
  "polarity_score": 0.45,
  "subjectivity": 0.62,
  "word_count": 6,
  "char_count": 44,
  "processing_time_ms": 12.3,
  "models": {
    "textblob":            { "prediction": "POSITIVE", "confidence": 95.0 },
    "naive_bayes":         { "prediction": "POSITIVE", "confidence": 89.3 },
    "logistic_regression": { "prediction": "POSITIVE", "confidence": 91.2 },
    "svm":                 { "prediction": "POSITIVE", "confidence": 93.1 }
  }
}
```

### `GET /health`

```json
{ "status": "ok", "models_loaded": 4 }
```

---

## 🛠 Troubleshooting

| Symptom | Cause & fix |
|---|---|
| Jupyter cell runs forever, no URL | Fixed via `use_reloader=False`. Use the provided `Run_Flask_App.ipynb`. |
| `FileNotFoundError: ...joblib` | Run from the project root, or re-train with `train_models.py`. Paths are now absolute, so `cd` into the project once and it works. |
| `ModuleNotFoundError` | Activate your venv and run `pip install -r requirements.txt`. |
| `Resource ... not found` (NLTK) | Allow the one-time NLTK download (needs internet) on first run. |
| Port 5000 already in use | `run_app(port=5050)` or `python flask_app.py` after freeing the port. |
| Browser can't reach the page | Confirm the terminal shows `Running on http://127.0.0.1:5000`. |

---

## 📜 License

Academic / educational use. Built as a final-year university project.
