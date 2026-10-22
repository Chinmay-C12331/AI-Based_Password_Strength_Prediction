#!/usr/bin/env python3
"""Data preparation script: cleans, deduplicates, and splits corpus."""

import os
import sys
import json
import argparse
from pwdstrength.utils import load_config, set_seed, setup_logger
from pwdstrength.data import (
    read_and_clean_corpus,
    split_data,
    cross_site_split,
    save_split_dataset,
    compute_data_summary,
)


def main():
    parser = argparse.ArgumentParser(description="Prepare, clean, deduplicate, and split password corpora")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--corpus", default="synthetic_demo", help="Name of corpus (file without extension in data/raw)")
    parser.add_argument("--train-corpus", default=None, help="Corpus A for cross-site training")
    parser.add_argument("--test-corpus", default=None, help="Corpus B for cross-site testing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)
    logger = setup_logger("prepare_data")

    raw_dir = cfg["paths"]["raw_data_dir"]
    proc_dir = cfg["paths"]["processed_data_dir"]
    res_dir = cfg["paths"]["results_dir"]
    os.makedirs(proc_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    max_clean_len = cfg.get("data", {}).get("max_clean_len", 64)

    # Cross-site mode
    if args.train_corpus and args.test_corpus:
        logger.info(f"Running in cross-site mode: Train={args.train_corpus} -> Test={args.test_corpus}")
        file_a = os.path.join(raw_dir, f"{args.train_corpus}.txt")
        file_b = os.path.join(raw_dir, f"{args.test_corpus}.txt")
        counts_a, stats_a = read_and_clean_corpus([file_a], max_clean_len=max_clean_len)
        counts_b, stats_b = read_and_clean_corpus([file_b], max_clean_len=max_clean_len)

        train, val, test = cross_site_split(counts_a, counts_b, seed=args.seed)
        out_folder = os.path.join(proc_dir, f"cross_{args.train_corpus}_to_{args.test_corpus}")
        save_split_dataset(train, os.path.join(out_folder, "train.parquet"))
        save_split_dataset(val, os.path.join(out_folder, "val.parquet"))
        save_split_dataset(test, os.path.join(out_folder, "test.parquet"))
        logger.info(f"Saved cross-site splits to {out_folder}: Train={len(train)}, Val={len(val)}, Test={len(test)}")
        return

    # In-distribution mode
    corpus_file = os.path.join(raw_dir, f"{args.corpus}.txt")
    if not os.path.exists(corpus_file):
        logger.warning(f"File {corpus_file} not found. Attempting to generate synthetic dataset...")
        from scripts.make_synthetic import generate_synthetic_passwords
        lines = generate_synthetic_passwords(target_unique=15000, seed=args.seed)
        os.makedirs(raw_dir, exist_ok=True)
        with open(corpus_file, "w", encoding="utf-8") as f:
            for l in lines:
                f.write(l + "\n")
        logger.info(f"Created fallback synthetic dataset at {corpus_file}")

    logger.info(f"Loading and cleaning corpus from {corpus_file}...")
    counts, drop_stats = read_and_clean_corpus([corpus_file], max_clean_len=max_clean_len)
    logger.info(f"Corpus cleaned: {drop_stats['unique_passwords_kept']} unique passwords kept.")

    train, val, test = split_data(
        counts,
        train_ratio=cfg["data"]["train_ratio"],
        val_ratio=cfg["data"]["val_ratio"],
        test_ratio=cfg["data"]["test_ratio"],
        seed=args.seed,
    )

    out_folder = os.path.join(proc_dir, args.corpus)
    save_split_dataset(train, os.path.join(out_folder, "train.parquet"))
    save_split_dataset(val, os.path.join(out_folder, "val.parquet"))
    save_split_dataset(test, os.path.join(out_folder, "test.parquet"))

    splits = {"train": train, "val": val, "test": test}
    summary = compute_data_summary(splits, drop_stats)
    summary_path = os.path.join(res_dir, "data_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Data summary saved to {summary_path}")
    print("\n--- DATASET SUMMARY TABLE ---")
    print(f"Total Lines Read: {drop_stats.get('total_lines_read', 0)}")
    print(f"Unique Passwords Kept: {drop_stats.get('unique_passwords_kept', 0)}")
    print(f"Train items: {len(train)} | Val items: {len(val)} | Test items: {len(test)}")
    print(f"Drops: {drop_stats}")
    print("------------------------------\n")


if __name__ == "__main__":
    main()
