"""Model training and evaluation loop."""

import os
import json
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score
from typing import Dict, Any, Tuple
from pwdstrength.model import HybridStrengthNet, AsymmetricHuberLoss
from pwdstrength.utils import set_seed, setup_logger


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_single_seed(
    model: HybridStrengthNet,
    train_tensors: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    val_tensors: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    seed: int,
    config: Dict[str, Any],
    run_name: str = "hybrid_model",
    max_epochs: int = 30,
    fast_mode: bool = False,
) -> Dict[str, Any]:
    """Train HybridStrengthNet or BiLSTM baseline for a single seed."""
    set_seed(seed)
    logger = setup_logger(f"train_seed_{seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    c_train, f_train, y_train, cls_train = [t.to(device) for t in train_tensors]
    c_val, f_val, y_val, cls_val = [t.to(device) for t in val_tensors]

    batch_size = 128 if fast_mode else config.get("training", {}).get("batch_size", 512)
    dataset = TensorDataset(c_train, f_train, y_train, cls_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    lr = config.get("training", {}).get("lr", 1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_cfg = config.get("loss", {})
    delta = loss_cfg.get("huber_delta", 1.0)
    alpha = loss_cfg.get("over_weight_alpha", 2.0) if loss_cfg.get("asymmetric", True) else 1.0
    lambda_ce = loss_cfg.get("lambda_ce", 0.5)
    criterion = AsymmetricHuberLoss(delta=delta, alpha=alpha, lambda_ce=lambda_ce)

    epochs = 3 if fast_mode else min(max_epochs, config.get("training", {}).get("max_epochs", 30))
    patience = config.get("training", {}).get("early_stop_patience", 3)
    grad_clip = config.get("training", {}).get("grad_clip", 1.0)

    best_val_mae = float("inf")
    patience_counter = 0
    best_state_dict = None

    logs = []
    logger.info(f"Training run={run_name} seed={seed} on {device} (epochs={epochs}, params={count_parameters(model):,})")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_acc = 0.0

        for b_c, b_f, b_y, b_cls in loader:
            optimizer.zero_grad()
            y_hat, logits = model(b_c, b_f)
            loss, _, _ = criterion(y_hat, b_y, logits, b_cls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_loss_acc += loss.item() * len(b_c)

        train_loss = train_loss_acc / len(c_train)

        # Validation pass
        model.eval()
        with torch.no_grad():
            val_y_hat, val_logits = model(c_val, f_val)
            val_loss, _, _ = criterion(val_y_hat, y_val, val_logits, cls_val)
            val_loss_val = val_loss.item()

            y_hat_np = val_y_hat.squeeze(-1).cpu().numpy()
            y_true_np = y_val.squeeze(-1).cpu().numpy()
            val_mae = float(np.mean(np.abs(y_hat_np - y_true_np)))

            cls_pred_np = torch.argmax(val_logits, dim=-1).cpu().numpy()
            cls_true_np = cls_val.cpu().numpy()
            val_f1 = float(f1_score(cls_true_np, cls_pred_np, average="macro", zero_division=0))

        logs.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss_val,
            "val_mae": val_mae,
            "val_macro_f1": val_f1,
        })
        logger.info(f"Epoch {epoch:2d}/{epochs}: TrainLoss={train_loss:.4f} ValMAE={val_mae:.4f} ValF1={val_f1:.4f}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

    # Save log CSV
    log_dir = os.path.join(config["paths"]["results_dir"], "train_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{run_name}_seed{seed}.csv")
    import pandas as pd
    pd.DataFrame(logs).to_csv(log_path, index=False)

    return {
        "best_val_mae": best_val_mae,
        "best_state_dict": best_state_dict,
        "param_count": count_parameters(model),
        "log_path": log_path,
    }
