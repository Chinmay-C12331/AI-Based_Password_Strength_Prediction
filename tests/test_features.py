"""Unit tests for pattern feature extractor and char encoder (Prompt 4)."""

import pytest
import numpy as np
from pwdstrength.features import (
    extract_features,
    encode_chars,
    FeatureScaler,
    compute_longest_repeat_run,
    compute_longest_sequential_run,
    compute_longest_keyboard_walk,
    FEATURE_NAMES,
)


def test_hand_crafted_feature_cases():
    # Test keyboard walk: "qwerty123" has "qwerty" length 6
    walk_len = compute_longest_keyboard_walk("qwerty123")
    assert walk_len == 6

    # Test repeat run: "aaaa1111" has run 4
    repeat_len = compute_longest_repeat_run("aaaa1111")
    assert repeat_len == 4

    # Test sequential run: "abcde" has 5, "12345" has 5
    seq_len = compute_longest_sequential_run("abcde")
    assert seq_len == 5

    # Test l33t changed: "P@ssw0rd" has @ and 0 -> changed = 1
    f_leet = extract_features("P@ssw0rd")
    # Feature 18 is leet_normalized_changed (index 17)
    assert f_leet[17] == 1.0

    # Test year detection: "Welcome2023" has 2023
    f_year = extract_features("Welcome2023")
    # Feature 15 is has_year (index 14)
    assert f_year[14] == 1.0

    # Test capitalized word: "Secret" has 1.0, "secret" has 0.0
    f_cap = extract_features("Secret")
    f_nocap = extract_features("secret")
    assert f_cap[21] == 1.0
    assert f_nocap[21] == 0.0


def test_char_encoding():
    # encode_chars should produce array of length 32
    enc = encode_chars("abc", max_len=32)
    assert len(enc) == 32
    assert enc[0] == ord("a") - 32 + 2
    assert enc[1] == ord("b") - 32 + 2
    assert enc[2] == ord("c") - 32 + 2
    # Padded with 0
    assert all(val == 0 for val in enc[3:])


def test_feature_scaler(tmp_path):
    X = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ], dtype=np.float32)

    scaler = FeatureScaler().fit(X)
    X_scaled = scaler.transform(X)

    # Scaled means should be ~0 and stds ~1
    assert np.allclose(np.mean(X_scaled, axis=0), 0.0, atol=1e-5)
    assert np.allclose(np.std(X_scaled, axis=0), 1.0, atol=1e-5)

    # Save and load round-trip
    p = tmp_path / "scaler.json"
    scaler.save(str(p))
    loaded = FeatureScaler().load(str(p))
    X_loaded_scaled = loaded.transform(X)
    assert np.allclose(X_scaled, X_loaded_scaled)
