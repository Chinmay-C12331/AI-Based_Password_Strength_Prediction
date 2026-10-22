"""Unit tests for reference attackers and out-of-fold label generation (Prompt 3)."""

import pytest
import numpy as np
from pwdstrength.attackers import DictionaryAttacker, MarkovAttacker
from pwdstrength.labels import generate_oof_labels_train, compute_reference_label


def test_dictionary_ranks_and_ties():
    # 3 passwords: "pwA" with count 100, "pwB" with count 50, "pwC" with count 50
    # pwA has rank 1
    # pwB and pwC tie at positions 2 and 3 -> mean rank 2.5
    data = [("pwA", 100), ("pwB", 50), ("pwC", 50)]
    att = DictionaryAttacker().fit(data)

    assert att.guess_number("pwA") == 1.0
    assert att.guess_number("pwB") == 2.5
    assert att.guess_number("pwC") == 2.5
    assert att.guess_number("unseen") is None


def test_markov_conditional_normalization():
    # Model conditional distribution should sum to 1.0 across vocabulary for any context
    data = [("abc", 5), ("abd", 2), ("cde", 3)]
    att = MarkovAttacker(order=3).fit(data)

    vocab = list(att.vocab)
    for test_ctx in ["ab", "cd", "z"]:
        prob_sum = sum(att._prob_char(test_ctx, ch) for ch in vocab)
        # Should sum to 1.0 within numerical precision
        assert abs(prob_sum - 1.0) < 1e-4


def test_monte_carlo_estimator_sanity():
    # Toy model on uniform single-character strings where exact guess order is known
    toy_data = [("a", 10), ("b", 10), ("c", 10)]
    att = MarkovAttacker(order=2).fit(toy_data)
    att.fit_monte_carlo(n_samples=5000, seed=42)

    # All 3 characters are equally likely, so guess numbers should be small (1..10)
    g_a = att.guess_number("a")
    g_b = att.guess_number("b")
    assert 1.0 <= g_a <= 20.0
    assert 1.0 <= g_b <= 20.0


def test_oof_leakage_prevention():
    # Project rule 2: assert that for every training password, the fitted attackers used for its label did not see it
    data = [(f"train_pw_{i}", 1) for i in range(20)]
    labeled = generate_oof_labels_train(data, n_folds=4, mc_samples=500, seed=42)

    assert len(labeled) == 20
    for item in labeled:
        # In OOF generation, each item was evaluated by a dictionary attacker trained on the other 3 folds.
        # Therefore, g_dict MUST be None because the item was not present in the dictionary of the remaining folds!
        assert item["g_dict"] is None, f"Leakage detected: {item['password']} was seen by dictionary attacker"
        assert item["y"] >= 0.0
        assert 0 <= item["cls"] <= 4
