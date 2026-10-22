"""Unit tests for data pipeline (Prompt 2): cleaning, deduplication, and splits."""

import pytest
from pwdstrength.data import clean_line, read_and_clean_corpus, split_data, cross_site_split


def test_clean_line_rules():
    # Empty line
    assert clean_line("")[2] == "empty_line"
    assert clean_line("   \n")[2] == "empty_line"

    # Password with count prefix
    pw, cnt, reason = clean_line("1500  mySecretPass123\n")
    assert pw == "mySecretPass123"
    assert cnt == 1500
    assert reason is None

    # Length > 64
    long_pw = "a" * 65
    assert clean_line(long_pw)[2] == "exceeds_max_len"

    # Exactly 64 chars
    valid_long = "a" * 64
    pw, cnt, reason = clean_line(valid_long)
    assert pw == valid_long
    assert reason is None

    # Email:password stripping
    pw, cnt, reason = clean_line("user.name+test@example.com:actualPassword99")
    assert pw == "actualPassword99"
    assert reason is None

    # Control char
    assert clean_line("pass\x00word")[2] == "contains_control_char"
    assert clean_line("pass\x1bword")[2] == "contains_control_char"


def test_deduplication_and_counts(tmp_path):
    f = tmp_path / "raw.txt"
    f.write_text("10 pass1\n5 pass1\npass2\n3 pass2\n", encoding="utf-8")

    counts, stats = read_and_clean_corpus([str(f)])
    assert counts["pass1"] == 15
    assert counts["pass2"] == 4
    assert stats["unique_passwords_kept"] == 2


def test_split_disjointness():
    # Construct 100 unique items
    counts = {f"password_{i}": i + 1 for i in range(100)}
    train, val, test = split_data(counts, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)

    train_set = {pw for pw, _ in train}
    val_set = {pw for pw, _ in val}
    test_set = {pw for pw, _ in test}

    assert len(train) == 80
    assert len(val) == 10
    assert len(test) == 10

    # Ensure mutually disjoint
    assert len(train_set.intersection(val_set)) == 0
    assert len(train_set.intersection(test_set)) == 0
    assert len(val_set.intersection(test_set)) == 0


def test_cross_site_exclusion():
    corpus_a = {"common1": 10, "common2": 5, "only_a": 2}
    corpus_b = {"common1": 20, "only_b1": 4, "only_b2": 1}

    train, val, test = cross_site_split(corpus_a, corpus_b, train_ratio=0.8, seed=42)
    train_set = {pw for pw, _ in train}
    val_set = {pw for pw, _ in val}
    test_set = {pw for pw, _ in test}

    # Test passwords from B must NOT appear in A's train split
    for pw in test_set:
        assert pw not in train_set
