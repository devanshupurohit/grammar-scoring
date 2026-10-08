# Grammar Scoring Engine for Spoken Data

A speech grammar-scoring pipeline for predicting a continuous grammar score from spoken English audio.

## Problem

Each audio sample contains approximately 45–60 seconds of spoken English. The task is to predict a grammar score from 0 to 5.

The training set contains 769 labeled samples. The competition test set contains 216 audio files, while the provided sample submission specifies the exact rows required for submission.

## Approach

The final selected approach is:

```text
Audio
  ↓
Faster-Whisper
  ↓
Transcript
  ├── Linguistic features
  └── TF-IDF word/bigram features
           ↓
       Feature fusion
           ↓
      Ridge regression
           ↓
     Grammar score 0–5
```

### Linguistic features

The pipeline extracts:

- word count
- character count
- sentence count
- unique word count
- type-token ratio
- average word length
- long-word ratio
- average sentence length
- sentence-length standard deviation
- words per second

### TF-IDF

Word and bigram TF-IDF features are generated using:

- `ngram_range=(1, 2)`
- `min_df=2`
- `max_df=0.95`
- sublinear TF
- maximum 20,000 features

### Regression

Ridge regression with `alpha=10.0` is used because the dataset is relatively small compared with the dimensionality of the text representation.

## Experiments

The notebook evaluates:

1. Mean baseline
2. Linguistic features + Ridge
3. TF-IDF + Ridge
4. Sentence embeddings + Ridge
5. Linguistic + TF-IDF + Ridge

The strongest validated model was the linguistic + TF-IDF Ridge model, with approximately:

- Mean 5-fold CV RMSE: 1.005
- Mean 5-fold CV Pearson: 0.582

The exact final training metrics and submission path are recorded by the notebook.

## Repository structure

```text
grammar_scoring/
├── data/
├── notebooks/
│   └── grammar_scoring.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── audio.py
│   ├── transcription.py
│   ├── features.py
│   ├── model.py
│   └── predict.py
├── outputs/
├── requirements.txt
└── README.md
```

## Running the final pipeline

Install dependencies:

```bash
pip install -r requirements.txt
```

From the project root:

```bash
python -m src.predict
```

The final submission is written to:

```text
outputs/submissions/submission.csv
```

## Reproducibility

Whisper transcripts and engineered features are cached under `outputs/`. If those cached files exist, the prediction script reuses them instead of retranscribing every audio file.

The competition test labels are never used.

## Research notebook

The notebook contains the full experimental process, including EDA, baselines, cross-validation, model comparisons, interpretation, limitations, and final submission generation.
