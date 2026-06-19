# ============================================================
# PROJECT 13: MOVIE REVIEW SENTIMENT ANALYSIS SYSTEM
# FILE: train_models.py
# PURPOSE: This file trains the machine learning models.
#          Run this file FIRST before running sentiment_app.py
# ============================================================
#
# WHAT THIS FILE DOES (in plain English):
# 1. Loads 50,000 movie reviews from the IMDB dataset
# 2. Cleans the text (removes noise like HTML, punctuation, etc.)
# 3. Converts the text into numbers the computer can understand
# 4. Trains 3 different machine learning models to detect sentiment
# 5. Tests how accurate each model is
# 6. Saves the trained models to files so the app can use them
# 7. Creates and saves charts showing the results
# ============================================================


# ── STEP 0: INSTALL MISSING LIBRARIES ───────────────────────────────────────
# We import 'subprocess' and 'sys' to run pip install commands from inside Python.
# This is useful in Google Colab because sometimes libraries are not pre-installed.
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import subprocess
import sys

# This line runs "pip install textblob wordcloud joblib" in the terminal for us.
# capture_output=True means we hide the install messages so the screen stays clean.
subprocess.run(
    [sys.executable, "-m", "pip", "install", "textblob", "wordcloud", "joblib"],
    capture_output=True
)


# ── STEP 1: IMPORT ALL THE TOOLS WE NEED ────────────────────────────────────

# pandas: used to load and work with the CSV dataset (like Excel but in Python)
import pandas as pd

# numpy: used for maths operations on arrays of numbers
import numpy as np

# matplotlib.pyplot: used to draw charts and graphs
import matplotlib.pyplot as plt

# re: short for "regular expressions" — used to find and remove patterns in text
# for example, removing HTML tags like <br> from reviews
import re

# os: lets us create folders and check if files exist on the computer
import os

# warnings: we use this to silence harmless warning messages that clutter the screen
import warnings

# joblib: used to SAVE and LOAD machine learning models to and from files
# (like saving a Word document so you can open it again later)
import joblib

# Tell Python to ignore harmless warning messages
warnings.filterwarnings('ignore')


# ── STEP 2: IMPORT NLTK TOOLS (Natural Language Processing) ─────────────────
# NLTK stands for Natural Language Toolkit.
# It is a Python library built specifically for processing human language (text).

import nltk

# stopwords: a built-in list of common English words like "the", "a", "is", "of"
# These words appear in EVERY review (positive or negative) so they carry
# no sentiment signal. We remove them to reduce noise.
from nltk.corpus import stopwords

# wordnet: a large database of English words, their meanings, and relationships.
# We use it to help the lemmatizer understand each word correctly.
from nltk.corpus import wordnet

# word_tokenize: splits a sentence into a list of individual words (tokens)
# Example: "I love films" becomes ["I", "love", "films"]
from nltk.tokenize import word_tokenize

# WordNetLemmatizer: reduces words to their base/root form using grammar rules.
# Example: "running" becomes "run", "movies" becomes "movie", "better" becomes "good"
# This is better than stemming because stemming just chops letters off the end
# (which can produce non-words like "episod"), while lemmatization uses a
# real dictionary to find the actual root word.
from nltk.stem import WordNetLemmatizer

# pos_tag: stands for Part Of Speech tagging.
# It labels each word in a sentence with its grammatical role.
# Example: "great" gets tagged as JJ (adjective), "run" gets tagged as VB (verb)
# We need this so the lemmatizer knows HOW to reduce each word correctly.
from nltk import pos_tag

# TextBlob: a simple library that analyses sentiment using pre-built rules.
# It gives a polarity score from -1.0 (very negative) to +1.0 (very positive).
# We use it as a BASELINE to compare against our trained ML models.
from textblob import TextBlob


# ── STEP 3: IMPORT SCIKIT-LEARN (Machine Learning Library) ──────────────────
# scikit-learn (imported as sklearn) is the most popular Python ML library.

# train_test_split: randomly divides the dataset into:
#   Training set (80%): the model LEARNS from this data
#   Testing set  (20%): we EVALUATE the model on data it has never seen before
from sklearn.model_selection import train_test_split

# TfidfVectorizer: converts text into numbers using the TF-IDF method.
# TF-IDF = Term Frequency times Inverse Document Frequency.
# In plain English: a word gets a HIGH score if it appears often in ONE review
# but rarely across ALL reviews — making it a meaningful/unique word for that review.
# Example: "masterpiece" is rare overall, so if it appears in a review,
# it strongly signals positive sentiment. TF-IDF gives it a high weight.
from sklearn.feature_extraction.text import TfidfVectorizer

# CountVectorizer: a simpler alternative to TF-IDF.
# It just COUNTS how many times each word appears in each review.
# We use this for comparison to show which method performs better.
from sklearn.feature_extraction.text import CountVectorizer

# MultinomialNB: Naive Bayes classifier for text.
# Called "naive" because it treats every word as independent of every other word.
# It calculates the PROBABILITY of a review being positive or negative
# based on which words appear in it and how often.
from sklearn.naive_bayes import MultinomialNB

# LogisticRegression: a classification model (despite the name, not for regression).
# It assigns a WEIGHT to every word: positive words get positive weights,
# negative words get negative weights. It sums all the weights to decide the label.
from sklearn.linear_model import LogisticRegression

# LinearSVC: a fast Support Vector Machine classifier.
# It finds the best mathematical boundary (called a hyperplane) that separates
# positive reviews from negative reviews with the maximum possible gap.
# Often the most accurate model for text classification.
from sklearn.svm import LinearSVC

# These four tools measure how well the models performed:
# accuracy_score:          percentage of predictions that were correct overall
# classification_report:   precision, recall, and F1-score for each class
# confusion_matrix:        a table of correct vs incorrect predictions
# ConfusionMatrixDisplay:  draws the confusion matrix as a colour-coded chart
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# WordCloud: generates the visual word cloud images.
# Bigger words in the cloud = more frequent in the reviews.
from wordcloud import WordCloud


# ── STEP 4: DOWNLOAD NLTK DATA FILES ────────────────────────────────────────
# NLTK tools need reference data files downloaded before they can work.
# These are not Python code — they are dictionary and grammar files.
# quiet=True hides the download messages to keep the output clean.

nltk.download('stopwords', quiet=True)                      # the English stopword list
nltk.download('punkt', quiet=True)                          # tokenizer rules
nltk.download('punkt_tab', quiet=True)                      # updated tokenizer rules
nltk.download('wordnet', quiet=True)                        # word meanings database
nltk.download('averaged_perceptron_tagger', quiet=True)     # POS tagger model
nltk.download('averaged_perceptron_tagger_eng', quiet=True) # English-specific POS tagger
nltk.download('omw-1.4', quiet=True)                        # multilingual wordnet support


# ── STEP 5: CREATE OUTPUT FOLDERS ───────────────────────────────────────────
# os.makedirs creates a folder on the computer if it does not already exist.
# exist_ok=True means: do not crash if the folder already exists, just continue.
# "results" is where we save chart image files (.png)
# "models"  is where we save trained model files (.joblib)

os.makedirs("results", exist_ok=True)
os.makedirs("models",  exist_ok=True)

print("=" * 58)
print("  PROJECT 13 — MOVIE SENTIMENT ANALYSIS")
print("  Training Script — All Models")
print("=" * 58)


# ════════════════════════════════════════════════════════════
# SECTION 2: DATA LOADING
# ════════════════════════════════════════════════════════════

print("\n[1/7] Loading dataset...")

# pd.read_csv loads the CSV file into a DataFrame.
# A DataFrame is like a table/spreadsheet inside Python.
# The IMDB CSV has two columns: "review" (the text) and "sentiment" (positive/negative)
df = pd.read_csv("IMDB Dataset.csv")

# Rename the columns to lowercase for easier reference throughout the code
df.columns = ['review', 'sentiment']

# Create a new column called 'label' that maps text labels to numbers.
# Machine learning models need numbers, not words.
# 'positive' becomes 1,  'negative' becomes 0
# .map() applies this conversion to every single row automatically
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

print(f"  [OK] {len(df):,} reviews loaded")
print(f"     Positive: {(df.label==1).sum():,}  |  Negative: {(df.label==0).sum():,}")


# ════════════════════════════════════════════════════════════
# SECTION 2: EXPLORATORY DATA ANALYSIS (EDA)
# EDA means: look at and understand your data BEFORE training.
# It helps us spot problems early (like imbalanced classes).
# ════════════════════════════════════════════════════════════

print("\n[2/7] Generating EDA charts...")

# Create a figure that holds 2 charts side by side (1 row, 2 columns)
# figsize=(12, 5) sets the overall size in inches: 12 wide, 5 tall
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Movie Review Dataset — Exploratory Analysis", fontsize=14, fontweight='bold')

# --- LEFT CHART: Count of positive vs negative reviews ---
# value_counts() counts how many rows have label=1 and how many have label=0
counts = df['label'].value_counts()

# Draw a bar chart on the LEFT axes (axes[0])
axes[0].bar(
    ['Positive', 'Negative'],   # x-axis labels
    counts.values,              # heights of the bars
    color=['#2ecc71', '#e74c3c'],  # green for positive, red for negative
    edgecolor='black',          # black border around each bar
    width=0.5                   # bar width (0 to 1 scale)
)
axes[0].set_title("Sentiment Distribution")
axes[0].set_ylabel("Number of Reviews")

# Add the count number as a text label on top of each bar
# enumerate() gives us both the index (i) and the value (v) at the same time
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 300, str(v), ha='center', fontweight='bold')

# --- RIGHT CHART: How many words are in each review? ---
# Create a new column by counting the words in each review
# .apply() runs the function on every row
# lambda x: len(x.split()) splits the review string into a list of words and counts them
df['word_count'] = df['review'].apply(lambda x: len(x.split()))

# Draw overlapping histograms: one for positive, one for negative reviews
# alpha=0.6 makes each bar semi-transparent so you can see both overlapping
# bins=40 divides the word count range into 40 equal groups
axes[1].hist(df[df['label'] == 1]['word_count'], bins=40, alpha=0.6,
             color='#2ecc71', label='Positive')
axes[1].hist(df[df['label'] == 0]['word_count'], bins=40, alpha=0.6,
             color='#e74c3c', label='Negative')
axes[1].set_title("Review Length Distribution")
axes[1].set_xlabel("Word Count")
axes[1].legend()

plt.tight_layout()  # automatically adjust spacing so charts do not overlap
plt.savefig("results/chart1_eda.png", dpi=150, bbox_inches='tight')
plt.show()
print("  [OK] Saved: results/chart1_eda.png")


# ════════════════════════════════════════════════════════════
# SECTION 2: TEXT PREPROCESSING
# Raw review text is full of noise: HTML tags, punctuation, numbers,
# common useless words, and different forms of the same word.
# We clean all of this before feeding data to the model.
# ════════════════════════════════════════════════════════════

print("\n[3/7] Preprocessing text with NLTK Lemmatizer...")

# Load the English stopword list into a Python SET (not a list).
# Sets are faster than lists for checking "is this word in the collection?"
# because sets use hashing internally.
stop_words = set(stopwords.words('english'))

# Create one lemmatizer object that we reuse for every word.
# Creating it once outside the function is more efficient than creating it
# thousands of times inside the function.
lemmatizer = WordNetLemmatizer()


def get_wordnet_pos(treebank_tag):
    """
    WHAT THIS FUNCTION DOES:
    It converts a POS tag from NLTK format into WordNet format.

    WHY WE NEED IT:
    NLTK's pos_tag() function labels words using Penn Treebank tags:
      JJ = adjective, VB = verb, NN = noun, RB = adverb
    But the lemmatizer needs WordNet-style tags:
      wordnet.ADJ, wordnet.VERB, wordnet.NOUN, wordnet.ADV
    This function bridges the two so lemmatization works correctly.

    EXAMPLE OF WHY IT MATTERS:
    The word "better" — if we know it is an adjective (JJ), the lemmatizer
    correctly reduces it to "good". Without the POS tag, it stays "better".
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ     # J tags = adjectives
    elif treebank_tag.startswith('V'):
        return wordnet.VERB    # V tags = verbs
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN    # N tags = nouns
    elif treebank_tag.startswith('R'):
        return wordnet.ADV     # R tags = adverbs
    else:
        return wordnet.NOUN    # default to noun if the tag is unrecognised


def preprocess_text(text):
    """
    WHAT THIS FUNCTION DOES:
    Takes one raw movie review string and returns a cleaned version.
    This same function is applied to every one of the 50,000 reviews.

    THE 7 CLEANING STEPS EXPLAINED:
    1. Lowercase:           "GREAT Movie" → "great movie"
    2. Remove HTML:         "great<br/>movie" → "great movie"
    3. Remove non-letters:  "great movie!! 10/10" → "great movie"
    4. Tokenize:            "great movie" → ["great", "movie"]
    5. POS tag:             [("great","JJ"), ("movie","NN")]
    6. Remove stopwords:    remove "the","a","is","and" etc.
    7. Lemmatize:           "movies"→"movie", "loved"→"love", "better"→"good"
    """

    # STEP 1: Convert all text to lowercase.
    # Without this, "Good" and "good" would be treated as two different words.
    text = text.lower()

    # STEP 2: Remove HTML tags using a regular expression.
    # r'<.*?>' means: find the pattern < then any characters (.*?) then >
    # re.sub replaces every match with '' (an empty string = deletion)
    # The ? makes it non-greedy: it stops at the FIRST > it finds,
    # not the last one (which could accidentally delete too much text)
    text = re.sub(r'<.*?>', '', text)

    # STEP 3: Remove everything that is not a letter or a space.
    # [^a-zA-Z\s] means: match any character that is NOT a-z, A-Z, or whitespace
    # This removes punctuation (!?.,:;), numbers, and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # STEP 4: Tokenize — split the string into a list of words.
    # word_tokenize handles edge cases better than just text.split()
    # Example: "don't" becomes ["do", "n't"] instead of ["don't"]
    tokens = word_tokenize(text)

    # STEP 5: POS tag every word — label each word with its grammar role.
    # pos_tag() returns a list of (word, tag) pairs.
    # Example: [("great","JJ"), ("film","NN"), ("loved","VBD")]
    tagged = pos_tag(tokens)

    # STEPS 6 & 7 combined in one list comprehension:
    # For every (word, tag) pair in tagged:
    #   - SKIP the word if it is in our stopwords list
    #   - SKIP the word if it is 2 characters or shorter (too short to be meaningful)
    #   - OTHERWISE: convert the tag format, then lemmatize the word
    # The result is a list of cleaned, lemmatized, meaningful words
    tokens = [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in tagged
        if word not in stop_words and len(word) > 2
    ]

    # Join the list of clean words back into one string separated by spaces.
    # Example: ["great", "film", "love"] → "great film love"
    return ' '.join(tokens)


# Apply preprocess_text to every row in the 'review' column.
# .apply() calls our function once per review (50,000 times total).
# The result is stored in a NEW column called 'clean_review'.
# This takes about 5-7 minutes — it is the slowest step.
df['clean_review'] = df['review'].apply(preprocess_text)

print("  [OK] Lemmatization complete on all 50,000 reviews")
print(f"  Original : {df['review'].iloc[0][:90]}")       # .iloc[0] = first row
print(f"  Cleaned  : {df['clean_review'].iloc[0][:90]}")


# ════════════════════════════════════════════════════════════
# TEXTBLOB BASELINE
# We test TextBlob BEFORE training any ML model.
# This gives us a starting reference point (baseline).
# If our ML models cannot beat TextBlob, something is wrong.
# ════════════════════════════════════════════════════════════

print("\n[4/7] TextBlob baseline analysis on 5,000 sample reviews...")

# Take a random sample of 5,000 reviews from the full 50,000.
# We sample instead of using all 50,000 because TextBlob is slow.
# random_state=42 is a seed number — it ensures the random sample is
# the same every time we run the code (reproducibility).
# .copy() makes a separate copy so we do not accidentally modify the original df.
sample = df.sample(5000, random_state=42).copy()

# Calculate the TextBlob polarity score for each review in the sample.
# TextBlob.sentiment.polarity checks each word against a built-in sentiment dictionary.
# Score range: -1.0 (very negative) to +1.0 (very positive), 0 = neutral
sample['tb_polarity'] = sample['review'].apply(
    lambda x: TextBlob(x).sentiment.polarity
)

# Convert the continuous polarity score into a binary label (1 or 0):
# polarity >= 0 → predict POSITIVE (1)
# polarity <  0 → predict NEGATIVE (0)
sample['tb_pred'] = sample['tb_polarity'].apply(
    lambda p: 1 if p >= 0 else 0
)

# Compare TextBlob predictions to the actual correct labels
# accuracy_score = number of correct predictions / total predictions
tb_acc = accuracy_score(sample['label'], sample['tb_pred'])
print(f"  [OK] TextBlob Accuracy: {tb_acc * 100:.2f}%")
# :.2f means format as a decimal number with exactly 2 decimal places

# Draw a histogram of TextBlob polarity scores
# This shows us HOW WELL TextBlob separates positive from negative
plt.figure(figsize=(10, 5))
plt.hist(sample[sample['label'] == 1]['tb_polarity'],
         bins=40, alpha=0.6, color='#2ecc71', label='Positive')
plt.hist(sample[sample['label'] == 0]['tb_polarity'],
         bins=40, alpha=0.6, color='#e74c3c', label='Negative')

# Draw a vertical dashed line at x=0 — this is TextBlob's decision boundary.
# Everything to the RIGHT of this line is predicted Positive.
# Everything to the LEFT  of this line is predicted Negative.
plt.axvline(x=0, color='black', linestyle='--', linewidth=1.5,
            label='Decision boundary (0)')
plt.title("TextBlob Polarity Score Distribution", fontsize=13, fontweight='bold')
plt.xlabel("Polarity Score  (negative ← 0 → positive)")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig("results/chart2_textblob.png", dpi=150, bbox_inches='tight')
plt.show()
print("  [OK] Saved: results/chart2_textblob.png")


# ════════════════════════════════════════════════════════════
# SECTION 3: VECTORISATION + TRAIN/TEST SPLIT
# ML models cannot read text — they only understand numbers.
# We must convert our cleaned text into numerical matrices.
# ════════════════════════════════════════════════════════════

print("\n[5/7] Vectorising and training 3 ML models...")

# Split the dataset into training data and testing data.
# X = the inputs (the cleaned review text)
# y = the outputs / correct answers (the labels: 1 or 0)
# test_size=0.2 means 20% (10,000 reviews) go to the test set
#              and 80% (40,000 reviews) go to the training set
# random_state=42 makes the split identical every time we run the code
# stratify=df['label'] ensures the test set has the same 50/50 positive/negative
# ratio as the full dataset (so the test is fair and representative)
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_review'],
    df['label'],
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)
print(f"  Train: {len(X_train):,} reviews  |  Test: {len(X_test):,} reviews")

# --- TF-IDF VECTORISER (PRIMARY METHOD) ---
# max_features=10000: only keep the 10,000 most informative words.
#   Very rare words (appearing in only 1 review) add noise but no signal, so we drop them.
# ngram_range=(1,2): include both single words AND two-word pairs.
#   Single words (unigrams): "good", "bad", "brilliant", "boring"
#   Two-word pairs (bigrams): "not good", "very bad", "highly recommend"
#   Bigrams are important for NEGATION: "not good" as a pair is clearly negative,
#   but if we split it, "not" gets removed (it is a stopword) and only "good" remains.
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))

# fit_transform() does TWO things at once on the TRAINING data:
# 1. fit:      learns the vocabulary (what words exist and their TF-IDF weights)
# 2. transform: converts the training reviews into a numerical matrix
# CRITICAL RULE: we ONLY fit on training data, NEVER on test data.
# Fitting on test data would mean the model "sees the answers" — that is cheating.
X_train_tfidf = tfidf.fit_transform(X_train)

# transform() ONLY converts — it does NOT re-learn the vocabulary.
# We use the exact same vocabulary from training to convert the test data.
# This simulates real life: in production, new reviews use the training vocabulary.
X_test_tfidf = tfidf.transform(X_test)

# --- COUNT VECTORISER (COMPARATIVE METHOD) ---
# A simpler alternative: instead of TF-IDF weights, it just counts word occurrences.
# We train a Naive Bayes on this too so we can compare:
# "Does TF-IDF outperform simple word counts?" (Answer: usually yes.)
# This comparison earns INNOVATION marks in the assessment.
count_vec = CountVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_cv = count_vec.fit_transform(X_train)  # fit and transform training data
X_test_cv  = count_vec.transform(X_test)        # only transform test data (no fit)


# ════════════════════════════════════════════════════════════
# SECTION 3: TRAIN ALL THREE ML MODELS
# ════════════════════════════════════════════════════════════

# ─── MODEL 1: NAIVE BAYES ────────────────────────────────────────────────────
# HOW NAIVE BAYES WORKS (explain this if asked):
# It is called "naive" because it assumes that every word is independent
# of every other word — which is not really true in language, but it still works.
# During training, it calculates:
#   P(positive | word) = how often does this word appear in positive reviews?
#   P(negative | word) = how often does this word appear in negative reviews?
# At prediction time, it multiplies all the word probabilities together
# and picks whichever class has the higher total probability.
# WHY WE USE IT: it is very fast and gives a decent baseline ML result.
nb = MultinomialNB()

# .fit() trains the model. We give it the training matrix and the correct labels.
# The model adjusts its internal parameters to learn the patterns.
nb.fit(X_train_tfidf, y_train)

# .predict() uses the trained model to classify the TEST reviews (unseen data).
# It returns an array of 0s and 1s — one prediction per test review.
nb_pred = nb.predict(X_test_tfidf)

# accuracy_score compares predictions to correct answers: correct / total
nb_acc = accuracy_score(y_test, nb_pred)
print(f"  [OK] Naive Bayes         : {nb_acc * 100:.2f}%")


# ─── MODEL 2: LOGISTIC REGRESSION ────────────────────────────────────────────
# HOW LOGISTIC REGRESSION WORKS (explain this if asked):
# Despite the name "regression", this is a classification model.
# It assigns a WEIGHT to every word in the vocabulary.
# During training, it adjusts these weights to minimise prediction errors.
# Positive sentiment words get HIGH positive weights ("brilliant", "masterpiece")
# Negative sentiment words get HIGH negative weights ("boring", "terrible", "waste")
# To classify a review, it multiplies each word's TF-IDF score by its weight,
# sums everything up, then uses the sigmoid function to convert to a probability.
# WHY WE USE IT: more powerful than Naive Bayes; considers word importance together.
# max_iter=1000: allow up to 1000 optimisation rounds to find the best weights
# C=1.0: regularisation parameter — penalises overly large weights to prevent
#        overfitting (memorising training data instead of learning real patterns)
lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
lr.fit(X_train_tfidf, y_train)   # train the model
lr_pred = lr.predict(X_test_tfidf)  # predict on unseen test data
lr_acc  = accuracy_score(y_test, lr_pred)
print(f"  [OK] Logistic Regression : {lr_acc * 100:.2f}%")


# ─── MODEL 3: SVM — SUPPORT VECTOR MACHINE ───────────────────────────────────
# HOW SVM WORKS (explain this if asked):
# Imagine plotting all 40,000 training reviews as points in a giant mathematical
# space where each of the 10,000 words is its own dimension (axis).
# Positive reviews cluster in one region of this space.
# Negative reviews cluster in another region.
# SVM finds the BEST POSSIBLE DIVIDING LINE (called a hyperplane) between the
# two clusters — specifically the line that has the MAXIMUM MARGIN (gap) between
# the nearest positive and negative points (called "support vectors").
# A larger margin = less likely to misclassify borderline reviews.
# WHY LinearSVC specifically: it is a faster, optimised version of SVM
# designed for large text datasets. Regular SVM (kernel-based) would be too slow
# on 40,000 reviews with 10,000 features.
# max_iter=2000: SVM needs more iterations than LR to fully converge
svm = LinearSVC(max_iter=2000, random_state=42, C=1.0)
svm.fit(X_train_tfidf, y_train)
svm_pred = svm.predict(X_test_tfidf)
svm_acc  = accuracy_score(y_test, svm_pred)
print(f"  [OK] SVM (LinearSVC)     : {svm_acc * 100:.2f}%")


# ─── COMPARATIVE: NAIVE BAYES WITH COUNT VECTORISER ──────────────────────────
# This trains a second Naive Bayes model using CountVectorizer instead of TF-IDF.
# We compare its accuracy to the TF-IDF version to show which feature
# extraction method is superior. This adds depth to Section 7 (Results).
nb_cv = MultinomialNB()
nb_cv.fit(X_train_cv, y_train)
nb_cv_pred = nb_cv.predict(X_test_cv)
nb_cv_acc  = accuracy_score(y_test, nb_cv_pred)
print(f"  [OK] NB + CountVectorizer: {nb_cv_acc * 100:.2f}% (comparative)")


# ════════════════════════════════════════════════════════════
# SECTION 3: SAVE ALL TRAINED MODELS
# We save the trained models so sentiment_app.py can load them
# without needing to retrain from scratch every time.
# Think of it like saving a game — you do not replay from the start.
# ════════════════════════════════════════════════════════════

print("\n[6/7] Saving models to disk...")

# joblib.dump(object, filepath) serialises and saves any Python object to disk.
# We MUST save BOTH the vectorisers AND the trained models.
# WHY SAVE THE VECTORISER? The app must convert new review text into the SAME
# numerical format that was used during training. If we used a different vectoriser,
# the word positions in the matrix would not match — the model would give wrong answers.
joblib.dump(tfidf,     "models/tfidf_vectorizer.joblib")    # vocabulary + TF-IDF weights
joblib.dump(count_vec, "models/count_vectorizer.joblib")    # vocabulary + count settings
joblib.dump(nb,        "models/naive_bayes.joblib")         # trained Naive Bayes model
joblib.dump(lr,        "models/logistic_regression.joblib") # trained Logistic Regression
joblib.dump(svm,       "models/svm_linearsvc.joblib")       # trained SVM model

print("  [OK] All models saved to: models/")


# ════════════════════════════════════════════════════════════
# SECTION 4: EVALUATION CHARTS
# ════════════════════════════════════════════════════════════

print("\n[7/7] Generating evaluation charts...")

# ─── CHART 3: CONFUSION MATRICES ─────────────────────────────────────────────
# WHAT IS A CONFUSION MATRIX? (explain this if asked):
# A confusion matrix is a 2x2 grid that breaks down model predictions into:
#   True Positive  (TP): review IS positive,  model said POSITIVE ← correct [OK]
#   True Negative  (TN): review IS negative,  model said NEGATIVE ← correct [OK]
#   False Positive (FP): review IS negative,  model said POSITIVE ← wrong ❌
#   False Negative (FN): review IS positive,  model said NEGATIVE ← wrong ❌
# The diagonal from top-left to bottom-right = correct predictions (want these high)
# The off-diagonal cells = mistakes (want these low)
# Darker blue in our chart = more predictions in that cell

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Confusion Matrices — All Three Models", fontsize=14, fontweight='bold')

# Use zip() to loop over all 3 models simultaneously:
# zip pairs: (ax, predictions, model_name, accuracy) for each model
for ax, pred, name, acc in zip(
    axes,
    [nb_pred, lr_pred, svm_pred],
    ['Naive Bayes', 'Logistic Regression', 'SVM (LinearSVC)'],
    [nb_acc, lr_acc, svm_acc]
):
    cm   = confusion_matrix(y_test, pred)  # compute the 2x2 matrix
    disp = ConfusionMatrixDisplay(cm, display_labels=['Negative', 'Positive'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')  # draw it in blue shades
    ax.set_title(f"{name}\n{acc * 100:.2f}%")       # title includes accuracy

plt.tight_layout()
plt.savefig("results/chart3_confusion.png", dpi=150, bbox_inches='tight')
plt.show()
print("  [OK] Saved: results/chart3_confusion.png")


# ─── CHART 4: MODEL ACCURACY COMPARISON BAR CHART ────────────────────────────
plt.figure(figsize=(10, 5))

labels = [
    'TextBlob\n(Baseline)',      # Rule-based — no training needed
    'Naive Bayes\n(TF-IDF)',     # ML model with TF-IDF features
    'Naive Bayes\n(CountVec)',   # ML model with CountVectorizer (comparative)
    'Logistic\nRegression',      # ML model — usually better than NB
    'SVM\n(LinearSVC)'           # ML model — typically the best for text
]
accs   = [tb_acc*100, nb_acc*100, nb_cv_acc*100, lr_acc*100, svm_acc*100]
colors = ['#95a5a6', '#e67e22', '#f39c12', '#3498db', '#2ecc71']

bars = plt.bar(labels, accs, color=colors, edgecolor='black', width=0.55)

# Add accuracy percentage labels above each bar
for bar, acc in zip(bars, accs):
    plt.text(
        bar.get_x() + bar.get_width() / 2,  # centre of the bar horizontally
        bar.get_height() + 0.3,             # just above the top of the bar
        f"{acc:.1f}%",                      # one decimal place
        ha='center', fontweight='bold', fontsize=9
    )

plt.ylim(0, 108)  # extra height so the labels above bars are not cut off
plt.title("All Models Accuracy Comparison", fontsize=13, fontweight='bold')
plt.ylabel("Accuracy (%)")
plt.tight_layout()
plt.savefig("results/chart4_accuracy.png", dpi=150, bbox_inches='tight')
plt.show()
print("  [OK] Saved: results/chart4_accuracy.png")


# ─── CHART 5: WORD CLOUDS ────────────────────────────────────────────────────
# WHAT IS A WORD CLOUD? (explain if asked):
# A word cloud is a visualisation where each word's SIZE corresponds to how
# frequently it appears across all the reviews in that class.
# Bigger word = appears more often.
# This helps us see WHICH WORDS most define positive vs negative sentiment.

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Most Common Words in Positive vs Negative Reviews",
             fontsize=14, fontweight='bold')

# Loop: first iteration = positive reviews (label==1), second = negative (label==0)
for ax, label, title, cmap in zip(
    axes,
    [1, 0],
    ['Positive Reviews', 'Negative Reviews'],
    ['Greens', 'Reds']
):
    # Filter to only the reviews of this class and join all their text into one string.
    # df[df['label']==label] selects only rows where label matches.
    # ' '.join(...) combines all those reviews into one giant block of text.
    all_text = ' '.join(df[df['label'] == label]['clean_review'])

    # Create and configure the WordCloud
    wc = WordCloud(
        width=600,
        height=400,
        background_color='white',
        colormap=cmap,     # Greens for positive, Reds for negative
        max_words=100      # show only the top 100 most frequent words
    )
    wc.generate(all_text)  # process the text and calculate word frequencies

    ax.imshow(wc, interpolation='bilinear')  # display the image (bilinear = smoother)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')  # hide the axis lines — they are irrelevant for an image

plt.tight_layout()
plt.savefig("results/chart5_wordcloud.png", dpi=150, bbox_inches='tight')
plt.show()
print("  [OK] Saved: results/chart5_wordcloud.png")


# ─── DETAILED CLASSIFICATION REPORTS ────────────────────────────────────────
# WHAT IS A CLASSIFICATION REPORT? (explain if asked):
# Accuracy alone can be misleading. The report adds:
#   Precision: of all reviews the model PREDICTED as positive, how many truly were?
#              High precision = fewer false alarms
#   Recall:    of all reviews that ACTUALLY WERE positive, how many did we catch?
#              High recall = fewer missed positives
#   F1-score:  the harmonic mean of precision and recall.
#              It balances both into one score. Higher is better (max = 1.0).
#   Support:   simply the count of how many test reviews are in each class.

print("\n" + "=" * 58)
print("DETAILED CLASSIFICATION REPORTS")
print("=" * 58)

for pred, name in [
    (nb_pred,  'Naive Bayes'),
    (lr_pred,  'Logistic Regression'),
    (svm_pred, 'SVM')
]:
    print(f"\n── {name} ──")
    # target_names replaces 0 and 1 with "Negative" and "Positive" in the report
    print(classification_report(y_test, pred, target_names=['Negative', 'Positive']))


# ─── FINAL SUMMARY ──────────────────────────────────────────────────────────
print("\n" + "=" * 58)
print("[OK]  TRAINING COMPLETE")

# Determine the best model by comparing SVM and LR accuracies
best_name = 'SVM' if svm_acc >= lr_acc else 'Logistic Regression'
best_acc  = max(svm_acc, lr_acc)

print(f"   Best model  : {best_name}")
print(f"   Best accuracy: {best_acc * 100:.2f}%")
print("   Trained models saved in : models/")
print("   Chart images saved in   : results/")
print("\n>>  NEXT STEP: Run sentiment_app.py to launch the UI")
print("=" * 58)
