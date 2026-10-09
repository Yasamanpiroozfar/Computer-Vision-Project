from collections.abc import Callable
from typing import Final

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.neighbors import NearestNeighbors

from cvproj_exc.config import Config

UNKNOWN_LABEL: Final[int] = -1
_EPS: Final[float] = 1e-12


def _validate_training_input(
    x_train: np.ndarray, y_train: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    x_train = np.asarray(x_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=int)
    if x_train.ndim != 2:
        raise ValueError("x_train must be a two-dimensional array.")
    if y_train.ndim != 1 or len(y_train) != len(x_train):
        raise ValueError("y_train must be one-dimensional and match x_train.")
    if len(x_train) == 0:
        raise ValueError("The training set must not be empty.")
    return x_train, y_train


def _validate_test_input(x_test: np.ndarray, n_features: int) -> np.ndarray:
    x_test = np.asarray(x_test, dtype=np.float64)
    if x_test.ndim != 2 or x_test.shape[1] != n_features:
        raise ValueError("x_test must be two-dimensional with the training feature size.")
    return x_test


def _nearest_neighbor_model(reference: np.ndarray):
    if len(reference) == 0:
        return None
    return NearestNeighbors(
        n_neighbors=1,
        algorithm="brute",
        metric="euclidean",
        n_jobs=-1,
    ).fit(reference)


def _normalized_known_margin(
    known_distance: np.ndarray, unknown_distance: np.ndarray
) -> np.ndarray:
    """Knownness score; larger values indicate a more likely known sample."""
    return (unknown_distance - known_distance) / np.maximum(
        unknown_distance + known_distance, _EPS
    )


def spl_training(
    x_train: np.ndarray, y_train: np.ndarray
) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Train the single-pseudo-label model."""
    x_train, y_train = _validate_training_input(x_train, y_train)

    known_mask = y_train != UNKNOWN_LABEL
    known_x = x_train[known_mask]
    known_y = y_train[known_mask]
    unknown_x = x_train[~known_mask]

    known_model = _nearest_neighbor_model(known_x)
    unknown_model = _nearest_neighbor_model(unknown_x)
    n_features = x_train.shape[1]

    def spl_predict_fn(x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_test = _validate_test_input(x_test, n_features)

        if known_model is None:
            return (
                np.full(len(x_test), UNKNOWN_LABEL, dtype=int),
                np.full(len(x_test), -np.inf, dtype=float),
            )

        known_distance, known_index = known_model.kneighbors(
            x_test, return_distance=True
        )
        known_distance = known_distance[:, 0]
        known_index = known_index[:, 0]
        y_pred = known_y[known_index].astype(int, copy=True)

        if unknown_model is None:
            return y_pred, -known_distance

        unknown_distance = unknown_model.kneighbors(
            x_test, return_distance=True
        )[0][:, 0]

        # Compare nearest known and unknown samples.
        y_pred[unknown_distance < known_distance] = UNKNOWN_LABEL
        y_score = _normalized_known_margin(known_distance, unknown_distance)
        return np.asarray(y_pred), np.asarray(y_score, dtype=float)

    return spl_predict_fn


def mpl_training(
    x_train: np.ndarray, y_train: np.ndarray
) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Train the multi-pseudo-label model."""
    x_train, y_train = _validate_training_input(x_train, y_train)
    n_features = x_train.shape[1]

    known_labels = np.unique(y_train[y_train != UNKNOWN_LABEL])
    known_centroids = np.asarray(
        [x_train[y_train == label].mean(axis=0) for label in known_labels],
        dtype=np.float64,
    )

    # Use each KUC sample as an individual pseudo class.
    unknown_centroids = np.asarray(
        x_train[y_train == UNKNOWN_LABEL], dtype=np.float64
    )

    def mpl_predict_fn(x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_test = _validate_test_input(x_test, n_features)

        if len(known_centroids) == 0:
            return (
                np.full(len(x_test), UNKNOWN_LABEL, dtype=int),
                np.full(len(x_test), -np.inf, dtype=float),
            )

        known_index, known_distance = pairwise_distances_argmin_min(
            x_test, known_centroids, metric="euclidean"
        )
        y_pred = known_labels[known_index].astype(int, copy=True)

        if len(unknown_centroids) == 0:
            return y_pred, -known_distance

        _, unknown_distance = pairwise_distances_argmin_min(
            x_test, unknown_centroids, metric="euclidean"
        )

        # Map the nearest unknown pseudo class to -1.
        y_pred[unknown_distance < known_distance] = UNKNOWN_LABEL
        y_score = _normalized_known_margin(known_distance, unknown_distance)
        return np.asarray(y_pred), np.asarray(y_score, dtype=float)

    return mpl_predict_fn


def load_challenge_train_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Load the challenge training data.

    Returns
    -------
    x : array, shape (n_samples, n_features). The feature vectors.
    y : array, shape (n_samples,). The corresponding labels of samples x.
    """
    df = pd.read_csv(Config.CHAL_TRAIN_DATA, header=None).values
    x = df[:, :-1]
    y = df[:, -1].astype(int)
    return x, y


def main():
    x_train, y_train = load_challenge_train_data()
    spl_predict_fn = spl_training(x_train, y_train)
    mpl_predict_fn = mpl_training(x_train, y_train)

    x_test = x_train[:50]
    for name, predict_fn in (("SPL", spl_predict_fn), ("MPL", mpl_predict_fn)):
        y_pred, y_score = predict_fn(x_test)
        print(
            "{}: predictions={}, scores={}, unknown_predictions={}".format(
                name,
                y_pred.shape,
                y_score.shape,
                int(np.sum(y_pred == UNKNOWN_LABEL)),
            )
        )


if __name__ == "__main__":
    main()
