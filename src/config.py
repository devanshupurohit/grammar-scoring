from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_CSV = DATA_DIR / "sample_submission.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
TRANSCRIPT_DIR = OUTPUT_DIR / "transcripts"
FEATURE_DIR = OUTPUT_DIR / "features"
SUBMISSION_DIR = OUTPUT_DIR / "submissions"

TRAIN_TRANSCRIPT_FILE = TRANSCRIPT_DIR / "train_transcripts.csv"
TEST_TRANSCRIPT_FILE = TRANSCRIPT_DIR / "test_transcripts.csv"
TRAIN_FEATURE_FILE = FEATURE_DIR / "train_features.csv"
TEST_FEATURE_FILE = FEATURE_DIR / "test_features.csv"

ASR_MODEL_SIZE = "base.en"
RIDGE_ALPHA = 10.0
RANDOM_STATE = 42

TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95
TFIDF_MAX_FEATURES = 20000

MIN_SCORE = 0.0
MAX_SCORE = 5.0

for directory in (
    OUTPUT_DIR,
    TRANSCRIPT_DIR,
    FEATURE_DIR,
    SUBMISSION_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)
