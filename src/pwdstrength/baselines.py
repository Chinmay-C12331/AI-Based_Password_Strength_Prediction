"""Baseline models B1 to B4 for password guessability prediction."""

import math
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from pwdstrength.attackers import MarkovAttacker
from pwdstrength.features import extract_features, FeatureScaler


DEFAULT_THRESHOLDS = [3.0, 6.0, 8.0, 10.0]


class Baseline:
    """Abstract base class for guessability baselines."""

    def predict_y(self, passwords: List[str]) -> np.ndarray:
        raise NotImplementedError

    def predict_class(self, passwords: List[str]) -> np.ndarray:
        raise NotImplementedError


class CompositionBaseline(Baseline):
    """B1: Composition score mapping (length, num_classes) to median y."""

    def __init__(self, thresholds: List[float] = DEFAULT_THRESHOLDS):
        self.thresholds = thresholds
        # Lookup table: length (0..32) x class_count (0..4)
        self.lookup = np.zeros((33, 5), dtype=np.float32)

    @staticmethod
    def _get_classes_count(pw: str) -> int:
        c = 0
        if any(ch.islower() for ch in pw): c += 1
        if any(ch.isupper() for ch in pw): c += 1
        if any(ch.isdigit() for ch in pw): c += 1
        if any(not ch.isalnum() for ch in pw): c += 1
        return max(1, c)

    def fit(self, passwords: List[str], y_labels: List[float]) -> "CompositionBaseline":
        cells: Dict[Tuple[int, int], List[float]] = {}
        for pw, y in zip(passwords, y_labels):
            l = min(len(pw), 32)
            c = self._get_classes_count(pw)
            cells.setdefault((l, c), []).append(y)

        # Compute medians and apply monotonicity smoothing
        for l in range(33):
            for c in range(1, 5):
                vals = cells.get((l, c), [])
                if vals:
                    self.lookup[l, c] = float(np.median(vals))
                else:
                    # Heuristic fallback if cell empty
                    self.lookup[l, c] = max(0.5, l * 0.45 * c * 0.35)

        # Ensure monotonicity with length and character classes
        for c in range(1, 5):
            for l in range(1, 33):
                if self.lookup[l, c] < self.lookup[l - 1, c]:
                    self.lookup[l, c] = self.lookup[l - 1, c]

        return self

    def predict_y(self, passwords: List[str]) -> np.ndarray:
        preds = []
        for pw in passwords:
            l = min(len(pw), 32)
            c = self._get_classes_count(pw)
            preds.append(float(self.lookup[l, c]))
        return np.array(preds, dtype=np.float32)

    def predict_class(self, passwords: List[str]) -> np.ndarray:
        y_preds = self.predict_y(passwords)
        return np.digitize(y_preds, self.thresholds).astype(np.int64)


class ZxcvbnBaseline(Baseline):
    """B2: zxcvbn model baseline."""

    def __init__(self, thresholds: List[float] = DEFAULT_THRESHOLDS):
        self.thresholds = thresholds

    def predict_y(self, passwords: List[str]) -> np.ndarray:
        import zxcvbn
        ys = []
        for pw in passwords:
            res = zxcvbn.zxcvbn(pw[:32])
            ys.append(float(res.get("guesses_log10", 0.0)))
        return np.array(ys, dtype=np.float32)

    def predict_class(self, passwords: List[str]) -> np.ndarray:
        import zxcvbn
        clss = []
        for pw in passwords:
            res = zxcvbn.zxcvbn(pw[:32])
            clss.append(int(res.get("score", 0)))
        return np.array(clss, dtype=np.int64)


class MarkovMCBaseline(Baseline):
    """B3: Markov Monte Carlo alone (no dictionary component)."""

    def __init__(self, thresholds: List[float] = DEFAULT_THRESHOLDS):
        self.thresholds = thresholds
        self.attacker = MarkovAttacker(order=4)

    def fit(self, train_tuples: List[Tuple[str, int]], mc_samples: int = 20000, seed: int = 42) -> "MarkovMCBaseline":
        self.attacker.fit(train_tuples).fit_monte_carlo(n_samples=mc_samples, seed=seed)
        return self

    def predict_y(self, passwords: List[str]) -> np.ndarray:
        ys = []
        for pw in passwords:
            g = self.attacker.guess_number(pw)
            ys.append(float(math.log10(max(1.0, g))))
        return np.array(ys, dtype=np.float32)

    def predict_class(self, passwords: List[str]) -> np.ndarray:
        ys = self.predict_y(passwords)
        return np.digitize(ys, self.thresholds).astype(np.int64)


class XGBoostBaseline(Baseline):
    """B4: Gradient boosted trees (XGBoost) trained on pattern features."""

    def __init__(self, thresholds: List[float] = DEFAULT_THRESHOLDS):
        self.thresholds = thresholds
        self.regressor = None
        self.classifier = None
        self.scaler = None
        self.top_blocklist = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        cls_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "XGBoostBaseline":
        try:
            import xgboost as xgb
            self.regressor = xgb.XGBRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.08, random_state=42, n_jobs=-1
            )
            self.classifier = xgb.XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.08, random_state=42, n_jobs=-1
            )
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
            self.regressor = HistGradientBoostingRegressor(max_iter=100, max_depth=5, random_state=42)
            self.classifier = HistGradientBoostingClassifier(max_iter=100, max_depth=5, random_state=42)

        self.regressor.fit(X_train, y_train)
        self.classifier.fit(X_train, cls_train)
        return self

    def predict_y_from_features(self, X: np.ndarray) -> np.ndarray:
        preds = self.regressor.predict(X)
        return np.clip(preds, 0.0, 16.0).astype(np.float32)

    def predict_class_from_features(self, X: np.ndarray) -> np.ndarray:
        clss = self.classifier.predict(X)
        return np.clip(clss, 0, 4).astype(np.int64)

    def predict_y(self, passwords: List[str]) -> np.ndarray:
        feats = np.array([extract_features(p, top_blocklist=self.top_blocklist) for p in passwords], dtype=np.float32)
        if self.scaler:
            feats = self.scaler.transform(feats)
        return self.predict_y_from_features(feats)

    def predict_class(self, passwords: List[str]) -> np.ndarray:
        feats = np.array([extract_features(p, top_blocklist=self.top_blocklist) for p in passwords], dtype=np.float32)
        if self.scaler:
            feats = self.scaler.transform(feats)
        return self.predict_class_from_features(feats)
