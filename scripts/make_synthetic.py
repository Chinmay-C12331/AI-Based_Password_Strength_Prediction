#!/usr/bin/env python3
"""Synthetic password corpus generator for smoke tests and pipeline verification.

WARNING:
*******************************************************************************
* WARNING: THIS SYNTHETIC CORPUS IS FOR SMOKE TESTS AND VERIFICATION ONLY.   *
* DO NOT REPORT RESULTS FROM SYNTHETIC DATA IN THE RESEARCH PAPER!            *
* ALL EMPIRICAL RESULTS IN THE PAPER MUST USE REAL, PUBLICLY AVAILABLE        *
* RESEARCH CORPORA (E.G. ROCKYOU, MYSPACE) DOWNLOADED BY THE RESEARCHER.     *
*******************************************************************************
"""

import os
import sys
import random
import argparse
from typing import List

# Predefined templates and vocabulary
WORDS = [
    "password", "welcome", "admin", "dragon", "master", "monkey", "shadow", "sunshine",
    "princess", "football", "baseball", "superman", "batman", "trustno1", "letmein",
    "mustang", "michael", "jessica", "charlie", "jordan", "harley", "robert", "daniel",
    "thomas", "chelsea", "hunter", "ranger", "starwars", "pokemon", "coffee", "cookie",
    "summer", "winter", "autumn", "spring", "secret", "matrix", "freedom", "warrior",
    "winner", "champion", "diamond", "silver", "golden", "crystal", "wizard", "thunder"
]

WALKS = [
    "123456", "12345678", "123456789", "qwerty", "qwertyuiop", "asdfgh", "zxcvbn",
    "1qaz2wsx", "zaq12wsx", "1234", "111111", "000000", "987654321", "654321"
]

SYMBOLS = ["!", "@", "#", "$", "%", "^", "&", "*", "?", "_", "-", "."]

LEET_MAP = {"a": "@", "o": "0", "e": "3", "i": "1", "l": "1", "s": "$", "t": "7"}


def apply_leet(word: str) -> str:
    res = []
    for c in word:
        if c.lower() in LEET_MAP and random.random() < 0.6:
            res.append(LEET_MAP[c.lower()])
        else:
            res.append(c)
    return "".join(res)


def generate_synthetic_passwords(target_unique: int = 25000, seed: int = 42) -> List[str]:
    random.seed(seed)
    unique_set = set()
    passwords = []

    # Category 1: Common walks and pins
    for w in WALKS:
        unique_set.add(w)
        for num in range(10):
            unique_set.add(f"{w}{num}")
            unique_set.add(f"{num}{w}")

    # Category 2: Word + digits / years (1970 - 2026)
    for word in WORDS:
        unique_set.add(word)
        unique_set.add(word.capitalize())
        unique_set.add(word.upper())
        for year in range(1975, 2026):
            unique_set.add(f"{word}{year}")
            unique_set.add(f"{word.capitalize()}{year}")
            unique_set.add(f"{word.capitalize()}{year}!")
            unique_set.add(f"{word}{year % 100:02d}")

    # Category 3: Word + Symbols + Digits
    for word in WORDS[:25]:
        for sym in SYMBOLS[:6]:
            for d in ["1", "12", "123", "99", "01", "2024"]:
                unique_set.add(f"{word}{sym}{d}")
                unique_set.add(f"{word.capitalize()}{sym}{d}")
                unique_set.add(f"{d}{sym}{word}")

    # Category 4: L33t substitutions
    for word in WORDS:
        leet = apply_leet(word)
        unique_set.add(leet)
        unique_set.add(f"{leet}123")
        unique_set.add(f"{leet}!")
        unique_set.add(f"{leet}2020")

    # Category 5: Multi-word passphrases
    for _ in range(3000):
        w1, w2 = random.sample(WORDS, 2)
        sep = random.choice(["-", "_", " ", ""])
        unique_set.add(f"{w1}{sep}{w2}")
        unique_set.add(f"{w1.capitalize()}{sep}{w2.capitalize()}{random.randint(1, 99)}")

    # Category 6: Random alphanumeric strings (high strength / high guessability)
    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*"
    while len(unique_set) < target_unique:
        length = random.randint(8, 20)
        pw = "".join(random.choice(chars) for _ in range(length))
        unique_set.add(pw)

    all_pws = list(unique_set)
    random.shuffle(all_pws)

    # Apply Zipf-like frequency counts: top items have high occurrence counts
    # count <space> password
    lines = []
    for rank, pw in enumerate(all_pws, start=1):
        # Zipfian count: C / rank^s
        if rank <= 50:
            count = int(10000 / (rank ** 0.8)) + 50
        elif rank <= 500:
            count = int(1000 / (rank ** 0.5)) + 10
        elif rank <= 3000:
            count = random.randint(2, 8)
        else:
            count = 1
        lines.append(f"{count} {pw}")

    return lines


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic password corpus for smoke tests")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config")
    parser.add_argument("--out", default="data/raw/synthetic_demo.txt", help="Output path")
    parser.add_argument("--count", type=int, default=25000, help="Number of unique passwords to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print(" LOUD WARNING: GENERATING SYNTHETIC PASSWORD CORPUS FOR SMOKE TESTING ONLY")
    print(" DO NOT REPORT RESULTS FROM SYNTHETIC DATA IN THE RESEARCH PAPER!")
    print(" ALWAYS USE REAL RESEARCH DATA (E.G. ROCKYOU) FOR PUBLISHED RESULTS.")
    print("=" * 80 + "\n")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    lines = generate_synthetic_passwords(target_unique=args.count, seed=args.seed)

    with open(args.out, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    print(f"Generated {len(lines)} synthetic password entries saved to {args.out}")


if __name__ == "__main__":
    main()
