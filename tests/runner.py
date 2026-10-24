#!/usr/bin/env python3
"""Standalone zero-dependency test runner using Python's standard unittest module."""

import os
import sys
import unittest

# Ensure src/ is on sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


class TestSmoke(unittest.TestCase):
    def test_package_import(self):
        import pwdstrength
        self.assertEqual(pwdstrength.__version__, "0.1.0")

    def test_config_loader(self):
        from pwdstrength.utils import load_config
        cfg = load_config("configs/default.yaml")
        self.assertIn("paths", cfg)
        self.assertEqual(cfg["data"]["max_len"], 32)

    def test_seeding(self):
        import random
        from pwdstrength.utils import set_seed
        set_seed(1234)
        r1 = random.random()
        set_seed(1234)
        r2 = random.random()
        self.assertEqual(r1, r2)


class TestData(unittest.TestCase):
    def test_clean_line_rules(self):
        from pwdstrength.data import clean_line
        self.assertEqual(clean_line("")[2], "empty_line")
        pw, cnt, reason = clean_line("1500  mySecretPass123\n")
        self.assertEqual(pw, "mySecretPass123")
        self.assertEqual(cnt, 1500)
        self.assertIsNone(reason)

    def test_split_disjointness(self):
        from pwdstrength.data import split_data
        counts = {f"password_{i}": i + 1 for i in range(100)}
        train, val, test = split_data(counts, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)
        train_set = {pw for pw, _ in train}
        val_set = {pw for pw, _ in val}
        test_set = {pw for pw, _ in test}
        self.assertEqual(len(train_set.intersection(val_set)), 0)
        self.assertEqual(len(train_set.intersection(test_set)), 0)


class TestAttackers(unittest.TestCase):
    def test_dictionary_ranks(self):
        from pwdstrength.attackers import DictionaryAttacker
        data = [("pwA", 100), ("pwB", 50), ("pwC", 50)]
        att = DictionaryAttacker().fit(data)
        self.assertEqual(att.guess_number("pwA"), 1.0)
        self.assertEqual(att.guess_number("pwB"), 2.5)

    def test_markov_normalization(self):
        from pwdstrength.attackers import MarkovAttacker
        data = [("abc", 5), ("abd", 2), ("cde", 3)]
        att = MarkovAttacker(order=3).fit(data)
        vocab = list(att.vocab)
        for test_ctx in ["ab", "cd", "z"]:
            prob_sum = sum(att._prob_char(test_ctx, ch) for ch in vocab)
            self.assertAlmostEqual(prob_sum, 1.0, places=3)


class TestFeatures(unittest.TestCase):
    def test_hand_crafted_cases(self):
        from pwdstrength.features import (
            compute_longest_keyboard_walk,
            compute_longest_repeat_run,
            compute_longest_sequential_run,
            extract_features,
            encode_chars,
        )
        self.assertEqual(compute_longest_keyboard_walk("qwerty123"), 6)
        self.assertEqual(compute_longest_repeat_run("aaaa1111"), 4)
        self.assertEqual(compute_longest_sequential_run("abcde"), 5)

        enc = encode_chars("abc", max_len=32)
        self.assertEqual(len(enc), 32)
        self.assertEqual(enc[0], ord("a") - 32 + 2)


class TestEvaluate(unittest.TestCase):
    def test_metrics(self):
        import numpy as np
        from pwdstrength.evaluate import compute_metrics
        cls_true = np.array([0, 1, 2, 3, 4])
        cls_pred = np.array([1, 1, 4, 3, 2])
        y_true = np.array([2.0, 4.0, 7.0, 9.0, 12.0])
        y_pred = np.array([3.5, 4.0, 10.5, 9.0, 8.0])
        m = compute_metrics(y_true, y_pred, cls_true, cls_pred)
        self.assertAlmostEqual(m["oer"], 40.0, places=2)
        self.assertAlmostEqual(m["oer_2"], 20.0, places=2)


def main():
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    suite.addTests(loader.loadTestsFromTestCase(TestSmoke))
    suite.addTests(loader.loadTestsFromTestCase(TestData))
    suite.addTests(loader.loadTestsFromTestCase(TestAttackers))
    suite.addTests(loader.loadTestsFromTestCase(TestFeatures))
    suite.addTests(loader.loadTestsFromTestCase(TestEvaluate))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
