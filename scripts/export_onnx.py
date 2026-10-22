#!/usr/bin/env python3
"""Export PyTorch HybridStrengthNet model to ONNX FP32 and dynamically quantized INT8."""

import os
import json
import argparse
import shutil
import numpy as np
import torch
from pwdstrength.utils import load_config, setup_logger
from pwdstrength.model import HybridStrengthNet


def main():
    parser = argparse.ArgumentParser(description="Export HybridStrengthNet to ONNX FP32 and INT8")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file")
    parser.add_argument("--run-name", default="hybrid_proposed", help="Run directory in models/")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger("export_onnx")

    export_dir = cfg["paths"]["export_dir"]
    os.makedirs(export_dir, exist_ok=True)

    # Initialize model
    model = HybridStrengthNet(
        vocab_size=cfg["model"]["vocab_size"],
        embed_dim=cfg["model"]["embed_dim"],
        lstm_layers=cfg["model"]["lstm_layers"],
        lstm_units=cfg["model"]["lstm_units"],
        feature_dim=22,
        feature_mlp=cfg["model"]["feature_mlp"],
        dense_units=cfg["model"]["dense"],
        dropout=0.0,  # inference mode
        use_features=True,
        use_attention=True,
    )
    model.eval()

    # If checkpoint exists, load it; otherwise use initialized weights
    ckpt_path = os.path.join(cfg["paths"]["models_dir"], args.run_name, "seed0.pt")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        logger.info(f"Loaded checkpoint from {ckpt_path}")
    else:
        logger.info("Using initialized model weights for export.")

    # 1. Export FP32 ONNX
    fp32_path = os.path.join(export_dir, "model.onnx")
    dummy_char_ids = torch.zeros(1, 32, dtype=torch.long)
    dummy_features = torch.zeros(1, 22, dtype=torch.float32)

    logger.info(f"Exporting ONNX FP32 model to {fp32_path}...")
    torch.onnx.export(
        model,
        (dummy_char_ids, dummy_features),
        fp32_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["char_ids", "features"],
        output_names=["y_hat", "class_logits"],
        dynamic_axes={
            "char_ids": {0: "batch_size"},
            "features": {0: "batch_size"},
            "y_hat": {0: "batch_size"},
            "class_logits": {0: "batch_size"},
        },
    )
    fp32_size_kb = os.path.getsize(fp32_path) / 1024.0
    logger.info(f"Exported FP32 ONNX ({fp32_size_kb:.1f} KB)")

    # 2. Dynamic INT8 Quantization
    int8_path = os.path.join(export_dir, "model.int8.onnx")
    logger.info(f"Quantizing to dynamic INT8 at {int8_path}...")
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(
            model_input=fp32_path,
            model_output=int8_path,
            weight_type=QuantType.QInt8,
        )
        int8_size_kb = os.path.getsize(int8_path) / 1024.0
        logger.info(f"Exported INT8 ONNX ({int8_size_kb:.1f} KB, compression: {fp32_size_kb/int8_size_kb:.2f}x)")
    except Exception as e:
        logger.warning(f"INT8 quantization error: {e}. Copying FP32 as fallback.")
        shutil.copyfile(fp32_path, int8_path)

    # 3. Export metadata artifacts: vocab.json, config.json
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for code in range(32, 127):
        vocab[chr(code)] = code - 32 + 2

    with open(os.path.join(export_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)

    export_config = {
        "max_len": 32,
        "feature_dim": 22,
        "class_thresholds": cfg.get("data", {}).get("class_thresholds", [3.0, 6.0, 8.0, 10.0]),
        "level_names": ["Very Weak", "Weak", "Fair", "Strong", "Very Strong"],
        "model_architecture": "HybridStrengthNet (BiLSTM + Attention + Feature MLP)",
    }
    with open(os.path.join(export_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(export_config, f, indent=2)

    # Sync to web/model/
    web_model_dir = os.path.join(cfg["paths"]["web_dir"], "model")
    os.makedirs(web_model_dir, exist_ok=True)
    for fname in ["model.onnx", "model.int8.onnx", "vocab.json", "config.json", "features_scaler.json"]:
        src_f = os.path.join(export_dir, fname)
        if os.path.exists(src_f):
            shutil.copy(src_f, os.path.join(web_model_dir, fname))

    logger.info(f"Export artifacts successfully synced to {web_model_dir}")


if __name__ == "__main__":
    main()
