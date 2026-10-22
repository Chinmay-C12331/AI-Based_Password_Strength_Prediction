#!/usr/bin/env python3
"""Script to generate out-of-fold reference attacker labels for train/val/test."""

import os
import json
import argparse
import numpy as np
import pandas as pd
from pwdstrength.utils import load_config, set_seed, setup_logger
from pwdstrength.data import load_split_dataset
from pwdstrength.labels import generate_oof_labels_train, generate_labels_fitted_on_train


def main():
    parser = argparse.ArgumentParser(description="Generate reference attacker labels")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--corpus", default="synthetic_demo", help="Corpus subdirectory in data/processed")
    parser.add_argument("--mc-samples", type=int, default=10000, help="Number of Monte Carlo samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)
    logger = setup_logger("make_labels")

    proc_dir = os.path.join(cfg["paths"]["processed_data_dir"], args.corpus)
    res_dir = cfg["paths"]["results_dir"]
    os.makedirs(res_dir, exist_ok=True)

    logger.info(f"Loading dataset splits from {proc_dir}...")
    train_data = load_split_dataset(os.path.join(proc_dir, "train"))
    val_data = load_split_dataset(os.path.join(proc_dir, "val"))
    test_data = load_split_dataset(os.path.join(proc_dir, "test"))

    logger.info(f"Loaded {len(train_data)} train, {len(val_data)} val, {len(test_data)} test items.")

    # 1. Training split: Out-Of-Fold (K=5)
    logger.info("Generating OUT-OF-FOLD labels for training split (K=5 folds)...")
    folds = cfg.get("data", {}).get("folds", 5)
    mc_samples = args.mc_samples
    thresholds = cfg.get("data", {}).get("class_thresholds", [3.0, 6.0, 8.0, 10.0])

    train_labeled = generate_oof_labels_train(
        train_data,
        n_folds=folds,
        mc_samples=mc_samples,
        thresholds=thresholds,
        seed=args.seed,
    )

    # 2. Validation and Test splits: fitted on FULL training split
    logger.info("Generating labels for val split (attackers fitted on full train split)...")
    val_labeled = generate_labels_fitted_on_train(
        val_data, train_data, mc_samples=mc_samples, thresholds=thresholds, seed=args.seed
    )

    logger.info("Generating labels for test split (attackers fitted on full train split)...")
    test_labeled = generate_labels_fitted_on_train(
        test_data, train_data, mc_samples=mc_samples, thresholds=thresholds, seed=args.seed
    )

    # Save to parquet/json/csv
    for name, records in [("labels_train", train_labeled), ("labels_val", val_labeled), ("labels_test", test_labeled)]:
        df = pd.DataFrame(records)
        parquet_p = os.path.join(proc_dir, f"{name}.parquet")
        json_p = os.path.join(proc_dir, f"{name}.json")
        csv_p = os.path.join(proc_dir, f"{name}.csv")
        try:
            df.to_parquet(parquet_p, index=False)
        except Exception:
            pass
        df.to_json(json_p, orient="records")
        df.to_csv(csv_p, index=False)
        logger.info(f"Saved {name} ({len(df)} records) to {proc_dir}")

    # Compute and save label statistics
    all_y = [r["y"] for r in train_labeled]
    all_cls = [r["cls"] for r in train_labeled]

    class_counts = {int(c): int(np.sum(np.array(all_cls) == c)) for c in range(5)}
    hist_y, bin_edges = np.histogram(all_y, bins=10)

    stats = {
        "num_train_labeled": len(train_labeled),
        "mean_y": float(np.mean(all_y)),
        "std_y": float(np.std(all_y)),
        "min_y": float(np.min(all_y)),
        "max_y": float(np.max(all_y)),
        "class_distribution": class_counts,
        "y_histogram": {
            "counts": [int(h) for h in hist_y],
            "bin_edges": [float(b) for b in bin_edges],
        },
    }

    stats_path = os.path.join(res_dir, "label_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Label stats saved to {stats_path}")
    print("\n--- LABEL STATS ---")
    print(f"Mean y: {stats['mean_y']:.2f} | Std y: {stats['std_y']:.2f} | Min: {stats['min_y']:.2f} | Max: {stats['max_y']:.2f}")
    print(f"Class distribution (0..4): {class_counts}")
    print("-------------------\n")


if __name__ == "__main__":
    main()
