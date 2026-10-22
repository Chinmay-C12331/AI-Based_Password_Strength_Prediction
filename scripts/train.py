#!/usr/bin/env python3
"""Main training script for HybridStrengthNet and Baselines."""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
from pwdstrength.utils import load_config, set_seed, setup_logger
from pwdstrength.model import HybridStrengthNet
from pwdstrength.train import train_single_seed, count_parameters


def main():
    parser = argparse.ArgumentParser(description="Train HybridStrengthNet and ablation variants")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file")
    parser.add_argument("--corpus", default="synthetic_demo", help="Corpus name in data/processed")
    parser.add_argument("--run-name", default="hybrid_proposed", help="Name of model run")
    parser.add_argument("--seed", type=int, default=None, help="Specific seed to run")
    parser.add_argument("--fast", action="store_true", help="Fast mode for smoke tests and rapid check")
    parser.add_argument("--no-features", action="store_true", help="B5 BiLSTM-only variant")
    parser.add_argument("--no-attention", action="store_true", help="No attention pooling ablation")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger("train")

    proc_dir = os.path.join(cfg["paths"]["processed_data_dir"], args.corpus)
    models_dir = os.path.join(cfg["paths"]["models_dir"], args.run_name)
    os.makedirs(models_dir, exist_ok=True)

    # Load data
    train_npz = np.load(os.path.join(proc_dir, "features_train.npz"))
    val_npz = np.load(os.path.join(proc_dir, "features_val.npz"))

    with open(os.path.join(proc_dir, "labels_train.json"), "r") as f:
        train_labels = json.load(f)
    with open(os.path.join(proc_dir, "labels_val.json"), "r") as f:
        val_labels = json.load(f)

    # Prepare tensors
    c_train = torch.tensor(train_npz["char_ids"], dtype=torch.long)
    f_train = torch.tensor(train_npz["features_scaled"], dtype=torch.float32)
    y_train = torch.tensor([r["y"] for r in train_labels], dtype=torch.float32).unsqueeze(-1)
    cls_train = torch.tensor([r["cls"] for r in train_labels], dtype=torch.long)

    c_val = torch.tensor(val_npz["char_ids"], dtype=torch.long)
    f_val = torch.tensor(val_npz["features_scaled"], dtype=torch.float32)
    y_val = torch.tensor([r["y"] for r in val_labels], dtype=torch.float32).unsqueeze(-1)
    cls_val = torch.tensor([r["cls"] for r in val_labels], dtype=torch.long)

    train_tensors = (c_train, f_train, y_train, cls_train)
    val_tensors = (c_val, f_val, y_val, cls_val)

    seeds = [args.seed] if args.seed is not None else cfg.get("training", {}).get("seeds", [0, 1, 2, 3, 4])
    if args.fast:
        seeds = seeds[:1]

    use_features = not args.no_features
    use_attention = not args.no_attention

    sample_model = HybridStrengthNet(
        vocab_size=cfg["model"]["vocab_size"],
        embed_dim=cfg["model"]["embed_dim"],
        lstm_layers=cfg["model"]["lstm_layers"],
        lstm_units=cfg["model"]["lstm_units"],
        feature_dim=f_train.shape[1],
        feature_mlp=cfg["model"]["feature_mlp"],
        dense_units=cfg["model"]["dense"],
        dropout=cfg["model"]["dropout"],
        use_features=use_features,
        use_attention=use_attention,
    )
    param_count = count_parameters(sample_model)
    logger.info(f"Initialized {args.run_name} (Params: {param_count:,} ~{param_count / 1e6:.2f}M)")

    # Save params.json
    with open(os.path.join(models_dir, "params.json"), "w") as f:
        json.dump({"run_name": args.run_name, "param_count": param_count, "config": cfg}, f, indent=2)

    for s in seeds:
        model = HybridStrengthNet(
            vocab_size=cfg["model"]["vocab_size"],
            embed_dim=cfg["model"]["embed_dim"],
            lstm_layers=cfg["model"]["lstm_layers"],
            lstm_units=cfg["model"]["lstm_units"],
            feature_dim=f_train.shape[1],
            feature_mlp=cfg["model"]["feature_mlp"],
            dense_units=cfg["model"]["dense"],
            dropout=cfg["model"]["dropout"],
            use_features=use_features,
            use_attention=use_attention,
        )

        res = train_single_seed(
            model=model,
            train_tensors=train_tensors,
            val_tensors=val_tensors,
            seed=s,
            config=cfg,
            run_name=args.run_name,
            fast_mode=args.fast,
        )

        ckpt_path = os.path.join(models_dir, f"seed{s}.pt")
        torch.save(res["best_state_dict"], ckpt_path)
        logger.info(f"Saved best checkpoint to {ckpt_path} (Val MAE={res['best_val_mae']:.4f})")

    # Copy scaler and config snapshot
    scaler_src = os.path.join(cfg["paths"]["models_dir"], "export", "features_scaler.json")
    if os.path.exists(scaler_src):
        import shutil
        shutil.copy(scaler_src, os.path.join(models_dir, "features_scaler.json"))


if __name__ == "__main__":
    main()
