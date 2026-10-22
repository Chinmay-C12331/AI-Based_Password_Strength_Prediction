"""Unit tests for Baselines B1 to B4 (Prompt 5)."""

import pytest
import numpy as np
from pwdstrength.baselines import (
    CompositionBaseline,
    ZxcvbnBaseline,
    MarkovMCBaseline,
    XGBoostBaseline,
)


def test_b1_composition():
    pws = ["12345", "Password123!", "abc", "SuperSecretPass2023!#"]
    y_labels = [2.0, 7.5, 1.5, 12.0]
    b1 = CompositionBaseline().fit(pws, y_labels)

    pred_y = b1.predict_y(pws)
    pred_cls = b1.predict_class(pws)

    assert len(pred_y) == len(pws)
    assert len(pred_cls) == len(pws)
    assert all(0 <= c <= 4 for c in pred_cls)
    # Monotonicity check: longer password should not have lower guess count than 3-char
    assert pred_y[3] >= pred_y[2]


def test_b2_zxcvbn():
    pws = ["123456", "correcthorsebatterystaple"]
    b2 = ZxcvbnBaseline()
    pred_y = b2.predict_y(pws)
    pred_cls = b2.predict_class(pws)

    assert len(pred_y) == 2
    assert len(pred_cls) == 2
    assert pred_cls[0] <= 1
    assert pred_cls[1] >= 2


def test_b3_markov_mc():
    train_tuples = [("abc", 10), ("xyz", 5), ("123", 20)]
    b3 = MarkovMCBaseline().fit(train_tuples, mc_samples=1000, seed=42)

    pred_y = b3.predict_y(["abc", "unseen_token_sequence"])
    pred_cls = b3.predict_class(["abc", "unseen_token_sequence"])

    assert len(pred_y) == 2
    assert len(pred_cls) == 2
    assert all(0 <= c <= 4 for c in pred_cls)


def test_b4_xgboost():
    # 10 samples with 22 features
    X = np.random.randn(10, 22).astype(np.float32)
    y = np.random.uniform(1.0, 12.0, size=(10,)).astype(np.float32)
    cls = np.random.randint(0, 5, size=(10,))

    b4 = XGBoostBaseline().fit(X, y, cls)
    pred_y = b4.predict_y_from_features(X)
    pred_cls = b4.predict_class_from_features(X)

    assert pred_y.shape == (10,)
    assert pred_cls.shape == (10,)
    assert all(0 <= c <= 4 for c in pred_cls)
