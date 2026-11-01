"""Decision-threshold tuning: one weight per class on the predicted probabilities.

The model is trained on the original (imbalanced) data. Its predicted
probabilities are multiplied by a per-class weight and the class with the
largest weighted probability wins. The weights are chosen by coordinate ascent
to maximise macro-F1 on the validation split; the test split is never used.
This costs nothing at training time and almost nothing at prediction time.
"""

import numpy as np

GRID = np.geomspace(0.1, 100, 22)


def macro_f1_codes(y, pred, n):
    """Macro-F1 over the classes present in y or pred, from integer codes (fast enough to call thousands of times)."""
    cm = np.bincount(y * n + pred, minlength=n * n).reshape(n, n)
    tp = np.diag(cm)
    denom = cm.sum(0) + cm.sum(1)
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    return f1[denom > 0].mean()


def tune_weights(proba, y_codes, rounds=3, grid=GRID):
    n = proba.shape[1]
    weights = np.ones(n)
    best = macro_f1_codes(y_codes, proba.argmax(1), n)
    for _ in range(rounds):
        improved = False
        for j in range(n):
            for g in grid:
                trial = weights.copy()
                trial[j] = g
                score = macro_f1_codes(y_codes, (proba * trial).argmax(1), n)
                if score > best + 1e-12:
                    best, weights, improved = score, trial, True
        if not improved:
            break
    return weights, best


class ThresholdedClassifier:
    """Wraps a fitted classifier; predicts the class with the largest weighted probability."""

    def __init__(self, model, weights):
        self.model = model
        self.weights = np.asarray(weights)
        self.classes_ = model.classes_

    def predict(self, X):
        return self.classes_[(self.model.predict_proba(X) * self.weights).argmax(1)]


def fit_thresholds(model, X_val, y_val):
    codes = {c: i for i, c in enumerate(model.classes_)}
    y_codes = np.array([codes[c] for c in y_val])
    weights, _ = tune_weights(model.predict_proba(X_val), y_codes)
    return ThresholdedClassifier(model, weights), dict(zip(map(str, model.classes_), np.round(weights, 3)))
