"""
Run the final grammar-scoring pipeline.

This script uses cached Whisper transcripts when available, trains the
selected Linguistic + TF-IDF Ridge model on all labeled training data,
predicts the competition test set, and writes a Kaggle-compatible submission.

Run from the project root:

    python -m src.predict
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    TRAIN_CSV,
    TEST_CSV,
    TRAIN_DIR,
    TEST_DIR,
    TRAIN_TRANSCRIPT_FILE,
    TEST_TRANSCRIPT_FILE,
    TRAIN_FEATURE_FILE,
    TEST_FEATURE_FILE,
    SUBMISSION_DIR,
    RIDGE_ALPHA,
)

from .features import (
    FEATURE_COLUMNS,
    build_linguistic_features,
    create_tfidf_vectorizer,
)

from .model import (
    build_final_model,
    prepare_combined_features,
    clip_predictions,
)

from .transcription import (
    load_whisper_model,
    transcribe_split,
)


def load_or_create_transcripts(train_df, test_df):
    """
    Load cached Whisper transcripts if available.

    Otherwise, generate transcripts for both train and test sets.
    """

    if (
        TRAIN_TRANSCRIPT_FILE.exists()
        and TEST_TRANSCRIPT_FILE.exists()
    ):
        train_transcripts = pd.read_csv(
            TRAIN_TRANSCRIPT_FILE
        )

        test_transcripts = pd.read_csv(
            TEST_TRANSCRIPT_FILE
        )

        print("Loaded cached transcripts.")

        return train_transcripts, test_transcripts

    print("Cached transcripts not found.")
    print("Generating transcripts with Faster-Whisper...")

    whisper = load_whisper_model()

    train_transcripts = transcribe_split(
        train_df,
        "train",
        TRAIN_DIR,
        TRAIN_TRANSCRIPT_FILE,
        model=whisper,
        resume=True,
    )

    test_transcripts = transcribe_split(
        test_df,
        "test",
        TEST_DIR,
        TEST_TRANSCRIPT_FILE,
        model=whisper,
        resume=True,
    )

    return train_transcripts, test_transcripts


def load_or_create_features(
    train_df,
    test_df,
    train_transcripts,
    test_transcripts,
):
    """
    Load cached linguistic features if available.

    Otherwise, calculate audio metadata and linguistic features
    for both train and test sets.
    """

    if (
        TRAIN_FEATURE_FILE.exists()
        and TEST_FEATURE_FILE.exists()
    ):
        train_features = pd.read_csv(
            TRAIN_FEATURE_FILE
        )

        test_features = pd.read_csv(
            TEST_FEATURE_FILE
        )

        print("Loaded cached linguistic features.")

        return train_features, test_features

    from .audio import get_audio_metadata

    def metadata_dataframe(df, audio_dir):
        rows = []

        for filename in df["filename"].astype(str):
            path = Path(audio_dir) / Path(filename).name

            metadata = get_audio_metadata(path)

            metadata["filename"] = filename

            rows.append(metadata)

        return pd.DataFrame(rows)

    train_metadata = metadata_dataframe(
        train_df,
        TRAIN_DIR,
    )

    test_metadata = metadata_dataframe(
        test_df,
        TEST_DIR,
    )

    train_features = build_linguistic_features(
        train_transcripts,
        train_metadata,
    )

    test_features = build_linguistic_features(
        test_transcripts,
        test_metadata,
    )

    train_features.to_csv(
        TRAIN_FEATURE_FILE,
        index=False,
    )

    test_features.to_csv(
        TEST_FEATURE_FILE,
        index=False,
    )

    return train_features, test_features


def build_submission(
    test_filenames,
    predictions,
):
    """
    Build the final submission using ALL rows in test.csv.

    Important:
        test.csv contains 216 rows in this competition, while
        sample_submission.csv contains only 204 rows.

    Therefore, sample_submission.csv must NOT be used to determine
    the number of rows in the final submission.

    The submission follows the filename and row order from test.csv.
    """

    test_filenames = [
        str(filename)
        for filename in test_filenames
    ]

    predictions = np.asarray(
        predictions,
        dtype=float,
    )

    # --------------------------------------------------------
    # Validate prediction count
    # --------------------------------------------------------

    if len(test_filenames) != len(predictions):
        raise RuntimeError(
            "Prediction count does not match test.csv. "
            f"test.csv has {len(test_filenames)} rows, "
            f"but there are {len(predictions)} predictions."
        )

    # --------------------------------------------------------
    # Build submission directly from test.csv
    # --------------------------------------------------------

    submission = pd.DataFrame(
        {
            "filename": test_filenames,
            "label": predictions,
        }
    )

    # --------------------------------------------------------
    # Clip predictions to valid grammar score range
    # --------------------------------------------------------

    submission["label"] = clip_predictions(
        submission["label"].astype(float)
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    if submission["filename"].isna().any():
        raise RuntimeError(
            "Submission contains missing filenames."
        )

    if submission["label"].isna().any():
        raise RuntimeError(
            "Submission contains missing predictions."
        )

    if not submission["label"].between(
        0,
        5,
    ).all():
        raise RuntimeError(
            "Submission contains predictions outside "
            "the valid 0-5 score range."
        )

    return submission[
        [
            "filename",
            "label",
        ]
    ]


def main():
    print("=" * 60)
    print("GRAMMAR SCORING - FINAL PIPELINE")
    print("=" * 60)

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    train_df = pd.read_csv(
        TRAIN_CSV
    )

    test_df = pd.read_csv(
        TEST_CSV
    )

    print(
        f"Training rows: {len(train_df)}"
    )

    print(
        f"Test rows: {len(test_df)}"
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Test labels are intentionally ignored.
    # --------------------------------------------------------

    y = (
        train_df["label"]
        .astype(float)
        .to_numpy()
    )

    # --------------------------------------------------------
    # Transcription
    # --------------------------------------------------------

    train_transcripts, test_transcripts = (
        load_or_create_transcripts(
            train_df,
            test_df,
        )
    )

    # --------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------

    train_features, test_features = (
        load_or_create_features(
            train_df,
            test_df,
            train_transcripts,
            test_transcripts,
        )
    )

    # --------------------------------------------------------
    # Attach training labels
    # --------------------------------------------------------

    train_feature_df = train_features.merge(
        train_df[
            [
                "filename",
                "label",
            ]
        ],
        on="filename",
        how="left",
    )

    test_feature_df = test_features.copy()

    # --------------------------------------------------------
    # Prepare transcript text
    # --------------------------------------------------------

    train_text = (
        train_transcripts["transcript"]
        .fillna("")
        .astype(str)
    )

    test_text = (
        test_transcripts["transcript"]
        .fillna("")
        .astype(str)
    )

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    vectorizer = create_tfidf_vectorizer()

    X_tfidf_train = vectorizer.fit_transform(
        train_text
    )

    X_tfidf_test = vectorizer.transform(
        test_text
    )

    # --------------------------------------------------------
    # Combine linguistic + TF-IDF features
    # --------------------------------------------------------

    X_train, scaler = prepare_combined_features(
        train_feature_df[
            FEATURE_COLUMNS
        ],
        X_tfidf_train,
        scaler=None,
        fit_scaler=True,
    )

    X_test, _ = prepare_combined_features(
        test_feature_df[
            FEATURE_COLUMNS
        ],
        X_tfidf_test,
        scaler=scaler,
        fit_scaler=False,
    )

    print(
        "Training matrix:",
        X_train.shape,
    )

    print(
        "Test matrix:",
        X_test.shape,
    )

    # --------------------------------------------------------
    # Train final Ridge model
    # --------------------------------------------------------

    model = build_final_model(
        alpha=RIDGE_ALPHA
    )

    model.fit(
        X_train,
        y,
    )

    # --------------------------------------------------------
    # Generate predictions
    # --------------------------------------------------------

    predictions = clip_predictions(
        model.predict(X_test)
    )

    print(
        "Predictions generated:",
        len(predictions),
    )

    print(
        "Prediction range:",
        predictions.min(),
        "-",
        predictions.max(),
    )

    # --------------------------------------------------------
    # Build final submission
    #
    # IMPORTANT:
    # Use ALL filenames from test.csv.
    # Do NOT use sample_submission.csv here.
    # --------------------------------------------------------

    submission = build_submission(
        test_df["filename"].astype(str).tolist(),
        predictions,
    )

    # --------------------------------------------------------
    # Final expected-size validation
    # --------------------------------------------------------

    if len(submission) != len(test_df):
        raise RuntimeError(
            "Submission size mismatch. "
            f"Expected {len(test_df)} rows, "
            f"got {len(submission)}."
        )

    # --------------------------------------------------------
    # Save submission
    # --------------------------------------------------------

    SUBMISSION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        SUBMISSION_DIR
        / "submission.csv"
    )

    submission.to_csv(
        output_file,
        index=False,
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("SUBMISSION CREATED")
    print("=" * 60)

    print(
        "Test rows:",
        len(test_df),
    )

    print(
        "Prediction rows:",
        len(predictions),
    )

    print(
        "Submission rows:",
        len(submission),
    )

    print(
        "Missing labels:",
        submission["label"].isna().sum(),
    )

    print(
        "Minimum:",
        submission["label"].min(),
    )

    print(
        "Maximum:",
        submission["label"].max(),
    )

    print(
        "Mean:",
        submission["label"].mean(),
    )

    print(
        "Saved to:",
        output_file,
    )

    print("\nFirst 5 rows:")
    print(
        submission.head()
    )

    return submission


if __name__ == "__main__":
    main()