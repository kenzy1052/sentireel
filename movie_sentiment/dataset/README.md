# Place the IMDB dataset here

This folder is intentionally empty in the delivered ZIP because the IMDB
dataset is large (~63 MB / 50,000 reviews).

## What to add

Copy your dataset file into this folder so the final path is exactly:

```
dataset/IMDB Dataset.csv
```

The CSV must contain two columns:

| column      | description                                  |
|-------------|----------------------------------------------|
| `review`    | the raw review text                          |
| `sentiment` | the label, either `positive` or `negative`   |

## Where to get it

Kaggle — "IMDB Dataset of 50K Movie Reviews":
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

## Do I need it to run the web app?

**No.** The web application (`flask_app.py`) only uses the pre-trained
`.joblib` models in `models/`. You only need this dataset if you want to
**re-train** the models with `train_models.py` or re-run the EDA notebook.
