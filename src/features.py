import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import (
    TFIDF_NGRAM_RANGE,
    TFIDF_MIN_DF,
    TFIDF_MAX_DF,
    TFIDF_MAX_FEATURES,
)

FEATURE_COLUMNS = [
    "word_count",
    "character_count",
    "sentence_count",
    "unique_word_count",
    "type_token_ratio",
    "average_word_length",
    "long_word_ratio",
    "average_sentence_length",
    "sentence_length_std",
    "words_per_second",
]


def extract_linguistic_features(transcript, duration):
    transcript = "" if pd.isna(transcript) else str(transcript).strip()

    words = re.findall(
        r"\b[a-zA-Z']+\b",
        transcript.lower(),
    )

    word_count = len(words)
    character_count = len(transcript)

    sentences = [
        sentence.strip()
        for sentence in re.split(r"[.!?]+", transcript)
        if sentence.strip()
    ]

    sentence_count = len(sentences)
    unique_word_count = len(set(words))

    type_token_ratio = (
        unique_word_count / word_count
        if word_count > 0
        else 0.0
    )

    average_word_length = (
        float(np.mean([len(word) for word in words]))
        if words
        else 0.0
    )

    long_word_ratio = (
        sum(len(word) >= 7 for word in words) / word_count
        if word_count > 0
        else 0.0
    )

    sentence_lengths = [
        len(re.findall(r"\b[a-zA-Z']+\b", sentence))
        for sentence in sentences
    ]

    average_sentence_length = (
        float(np.mean(sentence_lengths))
        if sentence_lengths
        else 0.0
    )

    sentence_length_std = (
        float(np.std(sentence_lengths))
        if len(sentence_lengths) > 1
        else 0.0
    )

    duration = float(duration) if pd.notna(duration) else 0.0

    words_per_second = (
        word_count / duration
        if duration > 0
        else 0.0
    )

    return {
        "word_count": word_count,
        "character_count": character_count,
        "sentence_count": sentence_count,
        "unique_word_count": unique_word_count,
        "type_token_ratio": type_token_ratio,
        "average_word_length": average_word_length,
        "long_word_ratio": long_word_ratio,
        "average_sentence_length": average_sentence_length,
        "sentence_length_std": sentence_length_std,
        "words_per_second": words_per_second,
    }


def build_linguistic_features(transcripts_df, metadata_df):
    merged = transcripts_df.merge(
        metadata_df[["filename", "duration"]],
        on="filename",
        how="left",
    )

    feature_rows = []

    for _, row in merged.iterrows():
        features = extract_linguistic_features(
            row["transcript"],
            row["duration"],
        )
        features["filename"] = row["filename"]
        feature_rows.append(features)

    return pd.DataFrame(feature_rows)


def create_tfidf_vectorizer():
    return TfidfVectorizer(
        lowercase=True,
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        sublinear_tf=True,
        max_features=TFIDF_MAX_FEATURES,
    )
