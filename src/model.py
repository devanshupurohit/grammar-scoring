import numpy as np
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .config import RIDGE_ALPHA, MIN_SCORE, MAX_SCORE


def build_final_model(alpha=RIDGE_ALPHA):
    return Ridge(alpha=alpha)


def prepare_combined_features(
    linguistic_features,
    tfidf_features,
    scaler=None,
    fit_scaler=True,
):
    if scaler is None:
        scaler = StandardScaler()

    if fit_scaler:
        linguistic_scaled = scaler.fit_transform(
            linguistic_features
        )
    else:
        linguistic_scaled = scaler.transform(
            linguistic_features
        )

    combined = sparse.hstack(
        [
            linguistic_scaled,
            tfidf_features,
        ],
        format="csr",
    )

    return combined, scaler


def clip_predictions(predictions):
    return np.clip(
        np.asarray(predictions, dtype=float),
        MIN_SCORE,
        MAX_SCORE,
    )
