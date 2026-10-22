"""Data loading, cleaning, deduplication, and splitting pipeline."""

import os
import re
import json
import random
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict


CONTROL_CHAR_REGEX = re.compile(r"[\x00-\x1f\x7f]")
EMAIL_PREFIX_REGEX = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+:(.+)$")


def clean_line(raw_line: str, max_clean_len: int = 64) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """Clean a single raw corpus line.

    Returns:
        (password, count, drop_reason)
        If valid: (password, count, None)
        If dropped: (None, None, reason)
    """
    line = raw_line.rstrip("\r\n")
    if not line:
        return None, None, "empty_line"

    # Check for count <space> password format
    count = 1
    # Check if leading tokens are integers (e.g. "   1234 password" or "1234 password")
    stripped = line.strip()
    match_count = re.match(r"^(\d+)\s+(.+)$", stripped)
    if match_count:
        count = int(match_count.group(1))
        line = match_count.group(2)
    else:
        line = stripped

    # Strip email if present (e.g. user@domain.com:password)
    email_match = EMAIL_PREFIX_REGEX.match(line)
    if email_match:
        line = email_match.group(1)

    # Strip trailing whitespace
    line = line.rstrip()

    if not line:
        return None, None, "empty_line"

    if len(line) > max_clean_len:
        return None, None, "exceeds_max_len"

    if CONTROL_CHAR_REGEX.search(line):
        return None, None, "contains_control_char"

    return line, count, None


def read_and_clean_corpus(
    filepaths: List[str], max_clean_len: int = 64
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Read one or more corpus files, clean and deduplicate while summing counts.

    Returns:
        (unique_passwords_dict, drop_stats)
    """
    counts: Dict[str, int] = defaultdict(int)
    drop_stats: Dict[str, int] = defaultdict(int)
    total_lines = 0

    for path in filepaths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Corpus file not found: {path}")

        # Try utf-8 first, fallback with errors='replace'
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                total_lines += 1
                pw, cnt, drop_reason = clean_line(line, max_clean_len=max_clean_len)
                if drop_reason:
                    drop_stats[drop_reason] += 1
                else:
                    counts[pw] += cnt

    drop_stats["total_lines_read"] = total_lines
    drop_stats["unique_passwords_kept"] = len(counts)
    return dict(counts), dict(drop_stats)


def split_data(
    counts: Dict[str, int],
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]]]:
    """Split UNIQUE passwords 80/10/10 with fixed seed ensuring zero overlap."""
    rng = random.Random(seed)
    unique_items = list(counts.items())
    rng.shuffle(unique_items)

    n = len(unique_items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = unique_items[:n_train]
    val = unique_items[n_train : n_train + n_val]
    test = unique_items[n_train + n_val :]

    return train, val, test


def cross_site_split(
    corpus_a_counts: Dict[str, int],
    corpus_b_counts: Dict[str, int],
    train_ratio: float = 0.80,
    seed: int = 42,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]]]:
    """Cross-site split: train/val come from Corpus A, test is all unique passwords from B not in A's train split."""
    train_a, val_a, _ = split_data(corpus_a_counts, train_ratio=train_ratio, val_ratio=1.0 - train_ratio, test_ratio=0.0, seed=seed)
    train_set = {pw for pw, _ in train_a}

    # Test is ALL unique passwords from B that do NOT appear in A's train split
    test_b = [(pw, cnt) for pw, cnt in corpus_b_counts.items() if pw not in train_set]

    return train_a, val_a, test_b


def save_split_dataset(
    data: List[Tuple[str, int]], out_path: str
) -> None:
    """Save dataset split to Parquet (if pyarrow/fastparquet exists) and JSON/CSV backup."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    base_no_ext = os.path.splitext(out_path)[0]

    # Save as JSON records
    records = [{"password": pw, "count": count} for pw, count in data]
    with open(f"{base_no_ext}.json", "w", encoding="utf-8") as f:
        json.dump(records, f)

    # Save as CSV
    with open(f"{base_no_ext}.csv", "w", encoding="utf-8") as f:
        f.write("password,count\n")
        for pw, cnt in data:
            safe_pw = pw.replace('"', '""')
            f.write(f'"{safe_pw}",{cnt}\n')

    # Try parquet via pandas/pyarrow if available
    try:
        import pandas as pd
        df = pd.DataFrame(records)
        df.to_parquet(f"{base_no_ext}.parquet", index=False)
    except Exception:
        pass


def load_split_dataset(path: str) -> List[Tuple[str, int]]:
    """Load dataset from parquet, json, or csv."""
    base_no_ext = os.path.splitext(path)[0]
    parquet_path = f"{base_no_ext}.parquet"
    json_path = f"{base_no_ext}.json"
    csv_path = f"{base_no_ext}.csv"

    if os.path.exists(parquet_path):
        try:
            import pandas as pd
            df = pd.read_parquet(parquet_path)
            return list(zip(df["password"], df["count"]))
        except Exception:
            pass

    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            items = json.load(f)
            return [(it["password"], it["count"]) for it in items]

    if os.path.exists(csv_path):
        results = []
        with open(csv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines[1:]:
                line = line.strip()
                if not line:
                    continue
                parts = line.rsplit(",", 1)
                if len(parts) == 2:
                    pw = parts[0].strip('"').replace('""', '"')
                    cnt = int(parts[1])
                    results.append((pw, cnt))
        return results

    raise FileNotFoundError(f"No split file found at {base_no_ext}.(parquet|json|csv)")


def compute_data_summary(
    splits: Dict[str, List[Tuple[str, int]]], drop_stats: Dict[str, int]
) -> Dict[str, Any]:
    """Compute summary statistics: length histogram, character-class mix, and drop reasons."""
    summary: Dict[str, Any] = {
        "drop_reasons": drop_stats,
        "split_counts": {},
        "length_histogram": defaultdict(int),
        "character_class_mix": {
            "has_lower": 0,
            "has_upper": 0,
            "has_digit": 0,
            "has_symbol": 0,
            "all_digits": 0,
            "all_letters": 0,
        },
    }

    total_unique = 0
    for split_name, items in splits.items():
        summary["split_counts"][split_name] = len(items)
        total_unique += len(items)
        for pw, _ in items:
            length_bucket = min(len(pw), 32)
            summary["length_histogram"][str(length_bucket)] += 1

            has_l = any(c.islower() for c in pw)
            has_u = any(c.isupper() for c in pw)
            has_d = any(c.isdigit() for c in pw)
            has_s = any(not c.isalnum() for c in pw)

            if has_l:
                summary["character_class_mix"]["has_lower"] += 1
            if has_u:
                summary["character_class_mix"]["has_upper"] += 1
            if has_d:
                summary["character_class_mix"]["has_digit"] += 1
            if has_s:
                summary["character_class_mix"]["has_symbol"] += 1
            if pw.isdigit():
                summary["character_class_mix"]["all_digits"] += 1
            if pw.isalpha():
                summary["character_class_mix"]["all_letters"] += 1

    summary["total_unique_passwords"] = total_unique
    return summary
