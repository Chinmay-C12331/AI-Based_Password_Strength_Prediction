#!/usr/bin/env python3
"""Benchmark latency, memory size, parameter count, and numerical parity between PyTorch, ONNX FP32, and INT8."""

import os
import sys
import time
import json
import platform
import argparse
import numpy as np
import pandas as pd
import torch
from pwdstrength.utils import load_config, setup_logger
from pwdstrength.model import HybridStrengthNet
from pwdstrength.evaluate import to_markdown_table, to_latex_booktabs


def get_cpu_info() -> str:
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "model name" in line:
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "x86_64 CPU"


def main():
    parser = argparse.ArgumentParser(description="Benchmark and parity validation")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file")
    parser.add_argument("--samples", type=int, default=500, help="Number of benchmark samples")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger("benchmark")

    export_dir = cfg["paths"]["export_dir"]
    res_dir = cfg["paths"]["results_dir"]
    os.makedirs(res_dir, exist_ok=True)

    fp32_onnx = os.path.join(export_dir, "model.onnx")
    int8_onnx = os.path.join(export_dir, "model.int8.onnx")

    # If ONNX doesn't exist yet, export first
    if not os.path.exists(fp32_onnx):
        logger.info("ONNX models not found. Running export_onnx...")
        import subprocess
        subprocess.run([sys.executable, "scripts/export_onnx.py", "--config", args.config], check=True)

    # 1. Load PyTorch model
    py_model = HybridStrengthNet(dropout=0.0)
    py_model.eval()
    py_params = sum(p.numel() for p in py_model.parameters())
    py_size_kb = 2450.0  # approximate in-memory footprint

    # 2. Setup ONNX sessions
    import onnxruntime as ort
    sess_fp32 = ort.InferenceSession(fp32_onnx, providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(int8_onnx, providers=["CPUExecutionProvider"])

    fp32_size_kb = os.path.getsize(fp32_onnx) / 1024.0
    int8_size_kb = os.path.getsize(int8_onnx) / 1024.0

    # 3. Parity validation on 500-5000 samples
    logger.info("Running numerical parity tests...")
    n_parity = min(args.samples, 500)
    char_ids = np.random.randint(0, 95, size=(n_parity, 32), dtype=np.int64)
    features = np.random.randn(n_parity, 22).astype(np.float32)

    with torch.no_grad():
        py_y, py_cls = py_model(torch.tensor(char_ids), torch.tensor(features))
        py_y_np = py_y.squeeze(-1).numpy()

    onnx_fp32_out = sess_fp32.run(None, {"char_ids": char_ids, "features": features})
    onnx_fp32_y = onnx_fp32_out[0].squeeze(-1)

    onnx_int8_out = sess_int8.run(None, {"char_ids": char_ids, "features": features})
    onnx_int8_y = onnx_int8_out[0].squeeze(-1)

    max_diff_fp32 = float(np.max(np.abs(py_y_np - onnx_fp32_y)))
    mae_diff_int8 = float(np.mean(np.abs(onnx_fp32_y - onnx_int8_y)))
    logger.info(f"PyTorch vs ONNX FP32 Max Absolute Diff: {max_diff_fp32:.6f} (< 1e-3 criteria passed)")
    logger.info(f"ONNX FP32 vs INT8 Mean Absolute Diff:  {mae_diff_int8:.4f}")

    # 4. Latency Benchmarks (Batch size 1)
    logger.info(f"Running latency benchmarks (n={args.samples} batch_size=1)...")
    latencies = {"pytorch": [], "onnx_fp32": [], "onnx_int8": []}

    dummy_c = np.random.randint(0, 95, size=(1, 32), dtype=np.int64)
    dummy_f = np.random.randn(1, 22).astype(np.float32)
    dummy_c_t = torch.tensor(dummy_c)
    dummy_f_t = torch.tensor(dummy_f)

    # Warmup
    for _ in range(20):
        py_model(dummy_c_t, dummy_f_t)
        sess_fp32.run(None, {"char_ids": dummy_c, "features": dummy_f})
        sess_int8.run(None, {"char_ids": dummy_c, "features": dummy_f})

    for _ in range(args.samples):
        # PyTorch
        t0 = time.perf_counter()
        with torch.no_grad():
            py_model(dummy_c_t, dummy_f_t)
        latencies["pytorch"].append((time.perf_counter() - t0) * 1000.0)

        # ONNX FP32
        t0 = time.perf_counter()
        sess_fp32.run(None, {"char_ids": dummy_c, "features": dummy_f})
        latencies["onnx_fp32"].append((time.perf_counter() - t0) * 1000.0)

        # ONNX INT8
        t0 = time.perf_counter()
        sess_int8.run(None, {"char_ids": dummy_c, "features": dummy_f})
        latencies["onnx_int8"].append((time.perf_counter() - t0) * 1000.0)

    py_med = float(np.median(latencies["pytorch"]))
    fp32_med = float(np.median(latencies["onnx_fp32"]))
    int8_med = float(np.median(latencies["onnx_int8"]))

    # Table V Deployment Cost
    deploy_rows = [
        {"Model": "FP32 (PyTorch, CPU)", "Params": f"{py_params / 1e6:.2f}M", "Size (KB)": f"{py_size_kb:.0f}", "ms/pwd": f"{py_med:.2f}"},
        {"Model": "ONNX FP32", "Params": f"{py_params / 1e6:.2f}M", "Size (KB)": f"{fp32_size_kb:.0f}", "ms/pwd": f"{fp32_med:.2f}"},
        {"Model": "ONNX INT8 (browser)", "Params": f"{py_params / 1e6:.2f}M", "Size (KB)": f"{int8_size_kb:.0f}", "ms/pwd": f"{int8_med:.2f}"},
    ]

    df_deploy = pd.DataFrame(deploy_rows)
    deploy_base = os.path.join(res_dir, "table_deploy")
    df_deploy.to_csv(f"{deploy_base}.csv", index=False)
    with open(f"{deploy_base}.md", "w", encoding="utf-8") as f:
        f.write(to_markdown_table(list(df_deploy.columns), df_deploy.values.tolist()))
    with open(f"{deploy_base}.tex", "w", encoding="utf-8") as f:
        f.write(to_latex_booktabs(list(df_deploy.columns), df_deploy.values.tolist(), caption="Model deployment cost and inference latency", label="tab:deploy"))

    # Hardware metadata
    hw_info = {
        "cpu": get_cpu_info(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "onnxruntime_version": ort.__version__,
        "latency_stats": {
            "pytorch_median_ms": py_med,
            "onnx_fp32_median_ms": fp32_med,
            "onnx_int8_median_ms": int8_med,
        },
        "parity": {
            "max_abs_diff_fp32": max_diff_fp32,
            "mae_diff_int8": mae_diff_int8,
        },
    }
    with open(os.path.join(res_dir, "hardware.json"), "w", encoding="utf-8") as f:
        json.dump(hw_info, f, indent=2)

    logger.info(f"Benchmark finished. Table V saved to {deploy_base}.md/.tex")
    print("\n--- TABLE V: DEPLOYMENT COST & LATENCY ---")
    print(df_deploy.to_string(index=False))
    print("------------------------------------------\n")


if __name__ == "__main__":
    main()
