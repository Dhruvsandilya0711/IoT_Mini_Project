"""Training-set rebalancing strategies. Never applied to validation or test data.

    none         original, imbalanced training data (the reproduced baseline)
    smote        SMOTE raises every class below --smote-target up to that size
    hybrid       random undersampling caps classes above --undersample-cap, then SMOTE as above
    classweight  no resampling; the model is trained with class_weight='balanced' instead

SMOTE only synthesises the classes that need it (a sampling_strategy dict),
so the minority classes are not blown up to the size of DDoS, which would not
fit in memory.
"""

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

STRATEGIES = ("none", "smote", "hybrid", "classweight")
DEFAULT_SMOTE_TARGET = 50_000
DEFAULT_UNDERSAMPLE_CAP = 200_000
DEFAULT_K_NEIGHBORS = 5


def smote_targets(y, target):
    return {cls: int(target) for cls, n in y.value_counts().items() if n < target}


def undersample_targets(y, cap):
    return {cls: int(cap) for cls, n in y.value_counts().items() if n > cap}


def resample(
    X, y, strategy,
    smote_target=DEFAULT_SMOTE_TARGET,
    undersample_cap=DEFAULT_UNDERSAMPLE_CAP,
    k_neighbors=DEFAULT_K_NEIGHBORS,
    seed=42,
):
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
    if strategy in ("none", "classweight"):
        return X, y

    if strategy == "hybrid":
        if undersample_cap < smote_target:
            raise ValueError("undersample_cap must be >= smote_target")
        under = undersample_targets(y, undersample_cap)
        if under:
            X, y = RandomUnderSampler(sampling_strategy=under, random_state=seed).fit_resample(X, y)

    over = smote_targets(y, smote_target)
    if over:
        smallest = int(y.value_counts()[list(over)].min())
        if smallest < 2:
            raise ValueError("SMOTE needs at least 2 training rows in every class it oversamples")
        k = min(k_neighbors, smallest - 1)
        if k < k_neighbors:
            print(f"Note: smallest class has {smallest} rows, using k_neighbors={k} instead of {k_neighbors}")
        X, y = SMOTE(sampling_strategy=over, k_neighbors=k, random_state=seed).fit_resample(X, y)
    return X, y
