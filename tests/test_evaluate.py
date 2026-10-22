"""Unit tests for evaluation metrics and formatting (Prompt 7)."""

import pytest
import numpy as np
from pwdstrength.evaluate import compute_metrics, to_markdown_table, to_latex_booktabs


def test_hand_computed_metrics():
    # 5 items:
    # true cls:  [0, 1, 2, 3, 4]
    # pred cls:  [1, 1, 4, 3, 2]
    # Item 0: pred > true (1 > 0) -> overestimate (+1)
    # Item 1: pred == true (1 == 1)
    # Item 2: pred > true (4 > 2, diff = 2) -> overestimate (+2, so OER-2 holds!)
    # Item 3: pred == true
    # Item 4: pred < true (2 < 4) -> underestimate
    # Overestimates: 2 out of 5 = 40.0%
    # Severe overestimates (>= +2): 1 out of 5 = 20.0%
    cls_true = np.array([0, 1, 2, 3, 4])
    cls_pred = np.array([1, 1, 4, 3, 2])

    y_true = np.array([2.0, 4.0, 7.0, 9.0, 12.0])
    y_pred = np.array([3.5, 4.0, 10.5, 9.0, 8.0])

    m = compute_metrics(y_true, y_pred, cls_true, cls_pred)

    assert abs(m["oer"] - 40.0) < 1e-4
    assert abs(m["oer_2"] - 20.0) < 1e-4
    # Check MAE: (|1.5| + |0| + |3.5| + |0| + |4.0|) / 5 = 9.0 / 5 = 1.8
    assert abs(m["mae"] - 1.8) < 1e-4


def test_markdown_and_latex_tables():
    headers = ["Method", "MAE", "OER %"]
    rows = [
        ["Proposed", "0.420", "4.6%"],
        ["zxcvbn", "1.150", "22.1%"],
    ]
    md = to_markdown_table(headers, rows)
    assert "| Method" in md
    assert "| Proposed" in md

    tex = to_latex_booktabs(headers, rows, caption="Test Table")
    assert "\\begin{table}" in tex
    assert "\\toprule" in tex
    assert "\\bottomrule" in tex
