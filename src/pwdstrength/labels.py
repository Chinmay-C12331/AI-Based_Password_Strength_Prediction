"""Out-of-fold (OOF) label generation using Dictionary and Markov MC reference attackers."""

import math
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from pwdstrength.attackers import DictionaryAttacker, MarkovAttacker


DEFAULT_THRESHOLDS = [3.0, 6.0, 8.0, 10.0]


def compute_reference_label(
    pw: str,
    dict_attacker: DictionaryAttacker,
    markov_attacker: MarkovAttacker,
    thresholds: List[float] = DEFAULT_THRESHOLDS,
) -> Tuple[float, int, Optional[float], float]:
    """Compute G*, y = log10(G*), and discrete strength class (0..4).

    Returns:
        (y, cls, g_dict, g_mc)
    """
    g_dict = dict_attacker.guess_number(pw)
    g_mc = markov_attacker.guess_number(pw)

    if g_dict is not None and g_dict > 0:
        g_star = min(g_dict, g_mc)
    else:
        g_star = g_mc

    g_star = max(1.0, g_star)
    y = float(math.log10(g_star))
    cls = int(np.digitize(y, thresholds))
    return y, cls, g_dict, g_mc


def generate_oof_labels_train(
    train_data: List[Tuple[str, int]],
    n_folds: int = 5,
    mc_samples: int = 50000,
    thresholds: List[float] = DEFAULT_THRESHOLDS,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generate training split labels OUT-OF-FOLD (Project Rule 2).

    K folds over unique passwords; the labels of fold k come from attackers fitted on the other K-1 folds.
    """
    n = len(train_data)
    fold_size = (n + n_folds - 1) // n_folds
    all_labeled_records: List[Dict[str, Any]] = []

    for fold_k in range(n_folds):
        start_idx = fold_k * fold_size
        end_idx = min(start_idx + fold_size, n)

        val_fold = train_data[start_idx:end_idx]
        train_fold = train_data[:start_idx] + train_data[end_idx:]

        # Fit attackers ONLY on other folds
        dict_att = DictionaryAttacker().fit(train_fold)
        markov_att = MarkovAttacker(order=4).fit(train_fold).fit_monte_carlo(n_samples=mc_samples, seed=seed + fold_k)

        for pw, cnt in val_fold:
            y, cls, g_dict, g_mc = compute_reference_label(pw, dict_att, markov_att, thresholds)
            all_labeled_records.append({
                "password": pw,
                "count": cnt,
                "y": y,
                "cls": cls,
                "g_dict": g_dict,
                "g_mc": g_mc,
            })

    return all_labeled_records


def generate_labels_fitted_on_train(
    query_data: List[Tuple[str, int]],
    full_train_data: List[Tuple[str, int]],
    mc_samples: int = 50000,
    thresholds: List[float] = DEFAULT_THRESHOLDS,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generate labels for validation and test splits from attackers fitted on the FULL training split."""
    dict_att = DictionaryAttacker().fit(full_train_data)
    markov_att = MarkovAttacker(order=4).fit(full_train_data).fit_monte_carlo(n_samples=mc_samples, seed=seed)

    records: List[Dict[str, Any]] = []
    for pw, cnt in query_data:
        y, cls, g_dict, g_mc = compute_reference_label(pw, dict_att, markov_att, thresholds)
        records.append({
            "password": pw,
            "count": cnt,
            "y": y,
            "cls": cls,
            "g_dict": g_dict,
            "g_mc": g_mc,
        })
    return records
