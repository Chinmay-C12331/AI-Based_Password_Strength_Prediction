#!/usr/bin/env python3
"""Run and evaluate Baselines B1 to B4, saving prediction parquets and CSVs."""

import os
import argparse
import numpy as np
import pandas as pd
from pwdstrength.utils import load_config, set_seed, setup_logger
from pwdstrength.data import load_split_dataset
from pwdstrength.baselines import (
    CompositionBaseline,
    ZxcvbnBaseline,
    MarkovMCBaseline,
    XGBoostBaseline,
)


def save_preds(df_preds: pd.DataFrame, out_path_base: str):
    os.makedirs(os.path.dirname(out_path_base), exist_ok=True)
    try:
        df_preds.to_parquet(f"{out_path_base}.parquet", index=False)
    except Exception:
        pass
    df_preds.to_csv(f"{out_path_base}.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Run Baselines B1-B4")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file")
    parser.add_argument("--corpus", default="synthetic_demo", help="Corpus subdirectory")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)
    logger = setup_logger("run_baselines")

    proc_dir = os.path.join(cfg["paths"]["processed_data_dir"], args.corpus)
    res_dir = cfg["paths"]["results_dir"]
    preds_dir = os.path.join(res_dir, "preds")
    os.makedirs(preds_dir, exist_ok=True)

    # Load labels
    train_labels_file = os.path.join(proc_dir, "labels_train.json")
    val_labels_file = os.path.join(proc_dir, "labels_val.json")
    test_labels_file = os.path.join(proc_dir, "labels_test.json")

    with open(train_labels_file, "r") as f:
        train_df = pd.DataFrame(json.load(f) if hasattr(json, "load") else pd.read_json(f))
    val_df = pd.read_json(val_labels_file)
    test_df = pd.read_json(test_labels_file)

    train_pws = train_df["password"].tolist()
    train_counts = train_df["count"].tolist()
    train_tuples = list(zip(train_pws, train_counts))

    val_pws = val_df["password"].tolist()
    test_pws = test_df["password"].tolist()

    # Load features if available
    train_npz = np.load(os.path.join(proc_dir, "features_train.npz"))
    val_npz = np.load(os.path.join(proc_dir, "features_val.npz"))
    test_npz = np.load(os.path.join(proc_dir, "features_test.npz"))

    X_train = train_npz["features_scaled"]
    X_val = val_npz["features_scaled"]
    X_test = test_npz["features_scaled"]

    y_train = train_df["y"].values
    cls_train = train_df["cls"].values

    thresholds = cfg.get("data", {}).get("class_thresholds", [3.0, 6.0, 8.0, 10.0])

    # 1. B1 Composition
    logger.info("Fitting and evaluating B1 Composition...")
    b1 = CompositionBaseline(thresholds=thresholds).fit(train_pws, y_train)
    for split_name, df_split, pws in [("val", val_df, val_pws), ("test", test_df, test_pws)]:
        y_pred = b1.predict_y(pws)
        cls_pred = b1.predict_class(pws)
        out_df = pd.DataFrame({
            "password": pws,
            "y_true": df_split["y"].values,
            "y_pred": y_pred,
            "cls_true": df_split["cls"].values,
            "cls_pred": cls_pred,
        })
        save_preds(out_df, os.path.join(preds_dir, f"b1_composition_{split_name}"))

    # 2. B2 zxcvbn
    logger.info("Evaluating B2 zxcvbn...")
    b2 = ZxcvbnBaseline(thresholds=thresholds)
    for split_name, df_split, pws in [("val", val_df, val_pws), ("test", test_df, test_pws)]:
        y_pred = b2.predict_y(pws)
        cls_pred = b2.predict_class(pws)
        out_df = pd.DataFrame({
            "password": pws,
            "y_true": df_split["y"].values,
            "y_pred": y_pred,
            "cls_true": df_split["cls"].values,
            "cls_pred": cls_pred,
        })
        save_preds(out_df, os.path.join(preds_dir, f"b2_zxcvbn_{split_name}"))

    # 3. B3 Markov MC alone
    logger.info("Fitting and evaluating B3 Markov MC...")
    b3 = MarkovMCBaseline(thresholds=thresholds).fit(train_tuples, mc_samples=10000, seed=args.seed)
    for split_name, df_split, pws in [("val", val_df, val_pws), ("test", test_df, test_pws)]:
        y_pred = b3.predict_y(pws)
        cls_pred = b3.predict_class(pws)
        out_df = pd.DataFrame({
            "password": pws,
            "y_true": df_split["y"].values,
            "y_pred": y_pred,
            "cls_true": df_split["cls"].values,
            "cls_pred": cls_pred,
        })
        save_preds(out_df, os.path.join(preds_dir, f"b3_markov_mc_{split_name}"))

    # 4. B4 XGBoost / GBDT on pattern features
    logger.info("Fitting and evaluating B4 XGBoost on pattern features...")
    b4 = XGBoostBaseline(thresholds=thresholds).fit(X_train, y_train, cls_train)
    for split_name, df_split, X_split in [("val", val_df, X_val), ("test", test_df, X_test)]:
        y_pred = b4.predict_y_from_features(X_split)
        cls_pred = b4.predict_class_from_features(X_split)
        out_df = pd.DataFrame({
            "password": df_split["password"].values,
            "y_true": df_split["y"].values,
            "y_pred": y_pred,
            "cls_true": df_split["cls"].values,
            "cls_pred": cls_pred,
        })
        save_preds(out_df, os.path.join(preds_dir, f"b4_xgboost_{split_name}"))

    logger.info(f"Baselines B1-B4 evaluations saved to {preds_dir}")


if __name__ == "__main__":
    import json
    main()
