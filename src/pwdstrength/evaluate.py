"""Evaluation metrics, tables generation, and publication-ready formatting."""

import math
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, f1_score, confusion_matrix
from typing import Dict, List, Tuple, Any, Optional


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cls_true: np.ndarray,
    cls_pred: np.ndarray,
) -> Dict[str, float]:
    """Calculate MAE, RMSE, Spearman rho, macro-F1, OER %, and OER-2 %."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))

    rho, _ = spearmanr(y_true, y_pred)
    if math.isnan(rho):
        rho = 0.0

    f1 = float(f1_score(cls_true, cls_pred, average="macro", zero_division=0))

    # Overestimation Rate (OER): cls_pred > cls_true
    oer = float(np.mean((cls_pred > cls_true).astype(float)) * 100.0)

    # Severe Overestimation Rate (OER-2): cls_pred >= cls_true + 2
    oer_2 = float(np.mean((cls_pred >= cls_true + 2).astype(float)) * 100.0)

    return {
        "mae": mae,
        "rmse": rmse,
        "rho": float(rho),
        "macro_f1": f1,
        "oer": oer,
        "oer_2": oer_2,
    }


def compute_confusion_matrix(cls_true: np.ndarray, cls_pred: np.ndarray) -> np.ndarray:
    """Compute 5x5 confusion matrix for strength classes 0..4."""
    return confusion_matrix(cls_true, cls_pred, labels=[0, 1, 2, 3, 4])


def format_val_std(mean: float, std: float, is_pct: bool = False, decimals: int = 3) -> str:
    """Format metric as 'mean ± std'."""
    if is_pct:
        return f"{mean:.1f} ± {std:.1f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def to_markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    """Convert list of row strings to Markdown table."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(str(val)))

    header_line = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    sep_line = "|-" + "-|-".join("-" * widths[i] for i in range(len(headers))) + "-|"

    row_lines = []
    for r in rows:
        line = "| " + " | ".join(str(val).ljust(widths[i]) for i, val in enumerate(r)) + " |"
        row_lines.append(line)

    return "\n".join([header_line, sep_line] + row_lines) + "\n"


def to_latex_booktabs(headers: List[str], rows: List[List[str]], caption: str = "", label: str = "") -> str:
    """Convert table to standard LaTeX booktabs format."""
    cols = "l" + "r" * (len(headers) - 1)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}" if caption else "",
        f"\\label{{{label}}}" if label else "",
        f"\\begin{{tabular}}{{{cols}}}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    for r in rows:
        lines.append(" & ".join(str(val) for val in r) + " \\\\")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    return "\n".join([l for l in lines if l]) + "\n"
