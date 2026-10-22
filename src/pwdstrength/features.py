"""Pattern-aware 22-dimensional feature extractor, character encoder, and feature scaler."""

import os
import re
import math
import json
from typing import List, Set, Dict, Optional, Any, Tuple
import numpy as np

# Feature Names list in exact order (1 to 22)
FEATURE_NAMES: List[str] = [
    "length",                            # 1
    "count_lower",                       # 2
    "count_upper",                       # 3
    "count_digit",                       # 4
    "count_symbol",                      # 5
    "ratio_lower",                       # 6
    "ratio_upper",                       # 7
    "ratio_digit",                       # 8
    "ratio_symbol",                      # 9
    "distinct_ratio",                    # 10
    "shannon_entropy",                   # 11
    "longest_repeat_run",                # 12
    "longest_sequential_run",            # 13
    "longest_keyboard_walk",             # 14
    "has_year",                          # 15
    "has_date",                          # 16
    "dict_word_coverage",                # 17
    "leet_normalized_changed",           # 18
    "is_in_training_top10k",             # 19
    "zxcvbn_guesses_log10",              # 20
    "trailing_digits_count",             # 21
    "is_capitalized_word",               # 22
]

# Standard QWERTY Layout Adjacency Map (lowercase)
QWERTY_ROWS = [
    "`1234567890-=",
    "qwertyuiop[]\\",
    "asdfghjkl;'",
    "zxcvbnm,./",
]
QWERTY_COLUMNS = [
    "1qaz", "2wsx", "3edc", "4rfv", "5tgb", "6yhn", "7ujm", "8ik,", "9ol.", "0p;/",
]

# Build adjacency pairs
KEYBOARD_ADJACENT: Set[Tuple[str, str]] = set()
for row in QWERTY_ROWS:
    for i in range(len(row) - 1):
        a, b = row[i], row[i + 1]
        KEYBOARD_ADJACENT.add((a, b))
        KEYBOARD_ADJACENT.add((b, a))
for col in QWERTY_COLUMNS:
    for i in range(len(col) - 1):
        a, b = col[i], col[i + 1]
        KEYBOARD_ADJACENT.add((a, b))
        KEYBOARD_ADJACENT.add((b, a))

# L33t normalization mapping
LEET_TABLE = str.maketrans({
    "@": "a",
    "0": "o",
    "1": "l",
    "3": "e",
    "$": "s",
    "5": "s",
    "7": "t",
})

YEAR_REGEX = re.compile(r"(?:19|20)\d{2}")
DATE_REGEX = re.compile(
    r"(?:\b(?:0[1-9]|[12]\d|3[01])(?:0[1-9]|1[0-2])(?:\d{2}|\d{4})\b)"
    r"|(?:\b(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])(?:\d{2}|\d{4})\b)"
)

# Global word list cache
_WORDLIST_CACHE: Optional[Set[str]] = None


def load_small_wordlist(wordlist_path: str = "data/wordlist_small.txt") -> Set[str]:
    """Load and cache small dictionary word list."""
    global _WORDLIST_CACHE
    if _WORDLIST_CACHE is not None:
        return _WORDLIST_CACHE

    words = set()
    if os.path.exists(wordlist_path):
        with open(wordlist_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                w = line.strip().lower()
                if w and len(w) >= 3 and not w.startswith("#"):
                    words.add(w)
    _WORDLIST_CACHE = words
    return words


def compute_longest_repeat_run(s: str) -> int:
    """Compute longest run of identical consecutive characters."""
    if not s:
        return 0
    max_run = 1
    cur_run = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            cur_run += 1
            if cur_run > max_run:
                max_run = cur_run
        else:
            cur_run = 1
    return max_run


def compute_longest_sequential_run(s: str) -> int:
    """Compute longest sequential run (e.g. abc, 123, cba, 321)."""
    if len(s) < 2:
        return len(s)

    max_len = 1
    cur_inc = 1
    cur_dec = 1

    lower_s = s.lower()
    for i in range(1, len(lower_s)):
        diff = ord(lower_s[i]) - ord(lower_s[i - 1])
        if diff == 1:
            cur_inc += 1
            cur_dec = 1
        elif diff == -1:
            cur_dec += 1
            cur_inc = 1
        else:
            cur_inc = 1
            cur_dec = 1
        max_len = max(max_len, cur_inc, cur_dec)

    return max_len


def compute_longest_keyboard_walk(s: str) -> int:
    """Compute longest keyboard walk on QWERTY layout in both directions."""
    if len(s) < 2:
        return len(s)

    lower_s = s.lower()
    max_walk = 1
    cur_walk = 1

    for i in range(1, len(lower_s)):
        pair = (lower_s[i - 1], lower_s[i])
        if pair in KEYBOARD_ADJACENT:
            cur_walk += 1
            if cur_walk > max_walk:
                max_walk = cur_walk
        else:
            cur_walk = 1

    return max_walk


def compute_dictionary_coverage(norm_pw: str, wordlist: Set[str]) -> float:
    """Compute fraction of password characters covered by dictionary words."""
    if not norm_pw or not wordlist:
        return 0.0

    n = len(norm_pw)
    covered = [False] * n

    for length in range(3, min(n + 1, 16)):
        for start in range(n - length + 1):
            sub = norm_pw[start : start + length]
            if sub in wordlist:
                for idx in range(start, start + length):
                    covered[idx] = True

    return sum(covered) / float(n)


def extract_features(
    pw: str,
    top_blocklist: Optional[Set[str]] = None,
    wordlist: Optional[Set[str]] = None,
    use_zxcvbn: bool = True,
) -> np.ndarray:
    """Extract 22-dimensional pattern feature vector from a password."""
    if wordlist is None:
        wordlist = load_small_wordlist()

    length = len(pw)
    if length == 0:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    # 1. length
    f1 = float(length)

    # 2-5. counts
    c_lower = sum(1 for c in pw if c.islower())
    c_upper = sum(1 for c in pw if c.isupper())
    c_digit = sum(1 for c in pw if c.isdigit())
    c_symbol = sum(1 for c in pw if not c.isalnum())

    f2 = float(c_lower)
    f3 = float(c_upper)
    f4 = float(c_digit)
    f5 = float(c_symbol)

    # 6-9. ratios
    f6 = c_lower / f1
    f7 = c_upper / f1
    f8 = c_digit / f1
    f9 = c_symbol / f1

    # 10. distinct ratio
    distinct_count = len(set(pw))
    f10 = distinct_count / f1

    # 11. Shannon entropy
    char_counts: Dict[str, int] = {}
    for c in pw:
        char_counts[c] = char_counts.get(c, 0) + 1
    entropy = -sum((cnt / f1) * math.log2(cnt / f1) for cnt in char_counts.values())
    f11 = float(entropy)

    # 12. longest repeat run
    f12 = float(compute_longest_repeat_run(pw))

    # 13. longest sequential run
    f13 = float(compute_longest_sequential_run(pw))

    # 14. longest keyboard walk
    f14 = float(compute_longest_keyboard_walk(pw))

    # 15. contains 4-digit year 1900-2099
    f15 = 1.0 if YEAR_REGEX.search(pw) else 0.0

    # 16. contains date pattern
    f16 = 1.0 if DATE_REGEX.search(pw) else 0.0

    # 17. dictionary word coverage after l33t normalization
    leet_norm = pw.translate(LEET_TABLE).lower()
    f17 = float(compute_dictionary_coverage(leet_norm, wordlist))

    # 18. l33t normalization changed password
    f18 = 1.0 if (leet_norm != pw.lower()) else 0.0

    # 19. in top-10k training blocklist
    f19 = 1.0 if (top_blocklist and pw in top_blocklist) else 0.0

    # 20. zxcvbn guesses_log10
    if use_zxcvbn:
        try:
            import zxcvbn
            res = zxcvbn.zxcvbn(pw[:32])
            f20 = float(res.get("guesses_log10", 0.0))
        except Exception:
            f20 = float(math.log10(max(1.0, float(distinct_count ** min(length, 12)))))
    else:
        f20 = 0.0

    # 21. trailing digits count
    trailing = 0
    for i in range(len(pw) - 1, -1, -1):
        if pw[i].isdigit():
            trailing += 1
        else:
            break
    f21 = float(trailing)

    # 22. is capitalized word pattern
    f22 = 1.0 if re.match(r"^[A-Z][a-z]+$", pw) else 0.0

    vec = np.array([
        f1, f2, f3, f4, f5, f6, f7, f8, f9, f10,
        f11, f12, f13, f14, f15, f16, f17, f18, f19, f20,
        f21, f22,
    ], dtype=np.float32)

    return vec


def encode_chars(pw: str, max_len: int = 32) -> np.ndarray:
    """Encode password string to token IDs [max_len].

    0 = pad, 1 = unknown, printable ASCII mapped to 2..96.
    """
    ids = np.zeros(max_len, dtype=np.int64)
    truncated = pw[:max_len]
    for i, ch in enumerate(truncated):
        code = ord(ch)
        if 32 <= code <= 126:
            ids[i] = code - 32 + 2
        else:
            ids[i] = 1  # unknown
    return ids


def get_feature_reasons(pw: str, feats: np.ndarray) -> List[str]:
    """Generate human-readable weakness or strength reasons based on feature values."""
    reasons: List[str] = []

    if feats[0] < 8:
        reasons.append("Password is too short (under 8 characters)")
    if feats[13] >= 4:  # keyboard walk
        reasons.append(f"Contains keyboard walk pattern of length {int(feats[13])}")
    if feats[14] > 0.5:  # year
        reasons.append("Contains a 4-digit birth/calendar year")
    if feats[16] >= 0.5:  # dict coverage
        reasons.append(f"Dictionary word detected ({int(feats[16] * 100)}% of password)")
    if feats[17] > 0.5:  # leet changed
        reasons.append("Predictable leetspeak substitution (@->a, 0->o, 3->e)")
    if feats[11] >= 3:  # repeat run
        reasons.append(f"Repeated character sequence (run of {int(feats[11])})")
    if feats[20] >= 2:  # trailing digits
        reasons.append("Appended simple trailing digits")
    if feats[18] > 0.5:  # blocklist
        reasons.append("Present in popular password blocklist")

    if not reasons:
        if feats[10] > 3.0 and feats[0] >= 12:
            reasons.append("High character entropy with mixed classes")
        else:
            reasons.append("Standard mixed alphanumeric composition")

    return reasons[:3]


class FeatureScaler:
    """Standardizes feature vectors using mean and standard deviation."""

    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "FeatureScaler":
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std = np.where(self.std < 1e-6, 1.0, self.std)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("FeatureScaler must be fitted before transform")
        return (X - self.mean) / self.std

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            "mean": self.mean.tolist() if self.mean is not None else [],
            "std": self.std.tolist() if self.std is not None else [],
            "feature_names": FEATURE_NAMES,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str) -> "FeatureScaler":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.mean = np.array(data["mean"], dtype=np.float32)
        self.std = np.array(data["std"], dtype=np.float32)
        return self
