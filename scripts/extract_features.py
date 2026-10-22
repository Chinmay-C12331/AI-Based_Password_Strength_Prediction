#!/usr/bin/env python3
"""Feature extraction script: extracts 22-d features, char IDs, fits FeatureScaler, and saves NPZ files."""

import os
import argparse
import numpy as np
import pandas as pd
from pwdstrength.utils import load_config, setup_logger
from pwdstrength.data import load_split_dataset
from pwdstrength.features import (
    extract_features,
    encode_chars,
    FeatureScaler,
    load_small_wordlist,
    FEATURE_NAMES,
)


def main():
    parser = argparse.ArgumentParser(description="Extract 22-dimensional pattern features for dataset splits")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--corpus", default="synthetic_demo", help="Corpus subdirectory in data/processed")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger("extract_features")

    proc_dir = os.path.join(cfg["paths"]["processed_data_dir"], args.corpus)
    max_len = cfg.get("data", {}).get("max_len", 32)
    use_zxcvbn = cfg.get("features", {}).get("use_zxcvbn_feature", True)
    top_k = cfg.get("features", {}).get("top_blocklist_k", 10000)

    # Load splits
    logger.info(f"Loading split data from {proc_dir}...")
    train_data = load_split_dataset(os.path.join(proc_dir, "train"))
    val_data = load_split_dataset(os.path.join(proc_dir, "val"))
    test_data = load_split_dataset(os.path.join(proc_dir, "test"))

    # Build top-10k blocklist from training split
    sorted_train = sorted(train_data, key=lambda x: x[1], reverse=True)
    top_blocklist = {pw for pw, _ in sorted_train[:top_k]}
    blocklist_file = os.path.join(proc_dir, "training_top10k.txt")
    with open(blocklist_file, "w", encoding="utf-8") as f:
        for pw, _ in sorted_train[:top_k]:
            f.write(f"{pw}\n")
    logger.info(f"Built and saved top-{len(top_blocklist)} training blocklist to {blocklist_file}")

    wordlist = load_small_wordlist(cfg["paths"]["wordlist_file"])

    splits = [
        ("train", train_data),
        ("val", val_data),
        ("test", test_data),
    ]

    raw_features = {}
    char_ids = {}

    for name, data in splits:
        logger.info(f"Extracting features for {name} split ({len(data)} items)...")
        feats_list = []
        cids_list = []
        for pw, _ in data:
            f = extract_features(
                pw,
                top_blocklist=top_blocklist,
                wordlist=wordlist,
                use_zxcvbn=use_zxcvbn,
            )
            c = encode_chars(pw, max_len=max_len)
            feats_list.append(f)
            cids_list.append(c)

        raw_features[name] = np.array(feats_list, dtype=np.float32)
        char_ids[name] = np.array(cids_list, dtype=np.int64)

    # Fit FeatureScaler on train
    scaler = FeatureScaler()
    scaler_path = os.path.join(cfg["paths"]["models_dir"], "export", "features_scaler.json")
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    scaler.fit(raw_features["train"])
    scaler.save(scaler_path)
    logger.info(f"Fitted FeatureScaler on train features and saved to {scaler_path}")

    # Scale and save NPZ for each split
    for name, data in splits:
        scaled_feat = scaler.transform(raw_features[name])
        c_ids = char_ids[name]
        pws = np.array([pw for pw, _ in data], dtype=object)
        out_npz = os.path.join(proc_dir, f"features_{name}.npz")
        np.savez_compressed(
            out_npz,
            features_raw=raw_features[name],
            features_scaled=scaled_feat,
            char_ids=c_ids,
            passwords=pws,
            feature_names=np.array(FEATURE_NAMES),
        )
        logger.info(f"Saved {out_npz} with features shape {scaled_feat.shape}")

    print("\n--- FEATURE EXTRACTION COMPLETED ---")
    print(f"Features dimension: {len(FEATURE_NAMES)}")
    print(f"Train features: {raw_features['train'].shape}")
    print(f"Val features:   {raw_features['val'].shape}")
    print(f"Test features:  {raw_features['test'].shape}")
    print("------------------------------------\n")


if __name__ == "__main__":
    main()
