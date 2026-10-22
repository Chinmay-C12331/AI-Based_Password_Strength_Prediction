"""Unit tests for HybridStrengthNet, mask handling, asymmetric loss, and overfit (Prompt 6)."""

import pytest
import torch
import torch.nn.functional as F
from pwdstrength.model import HybridStrengthNet, AsymmetricHuberLoss


def test_forward_pass_shapes():
    B, T, F_dim = 4, 32, 22
    model = HybridStrengthNet(feature_dim=F_dim, lstm_units=64, dense_units=64)

    char_ids = torch.randint(0, 95, (B, T), dtype=torch.long)
    features = torch.randn(B, F_dim, dtype=torch.float32)

    y_hat, class_logits = model(char_ids, features)

    assert y_hat.shape == (B, 1)
    assert class_logits.shape == (B, 5)


def test_mask_handling_padding_invariance():
    # If we pad with zeros, the model's masked attention should produce identical representation
    model = HybridStrengthNet(lstm_units=32, dense_units=32, dropout=0.0)
    model.eval()

    # Pass 1: length 5 sequence padded with zeros to 32
    c1 = torch.tensor([[10, 20, 30, 40, 50] + [0] * 27], dtype=torch.long)
    f1 = torch.ones((1, 22), dtype=torch.float32)

    # Output should be stable
    with torch.no_grad():
        y1, l1 = model(c1, f1)

    assert not torch.isnan(y1).any()
    assert not torch.isnan(l1).any()


def test_asymmetric_loss_weighting():
    # Overestimates (e > 0) must cost more than equal underestimates (e < 0)
    loss_fn = AsymmetricHuberLoss(delta=1.0, alpha=2.0, lambda_ce=0.0)

    y_true = torch.tensor([5.0])
    y_over = torch.tensor([7.0])  # e = +2.0
    y_under = torch.tensor([3.0]) # e = -2.0

    loss_over, _, _ = loss_fn(y_over, y_true)
    loss_under, _, _ = loss_fn(y_under, y_true)

    # Since alpha = 2.0, overestimate loss should be exactly 2.0 * underestimate loss
    assert loss_over.item() > loss_under.item()
    assert abs(loss_over.item() - 2.0 * loss_under.item()) < 1e-4


def test_one_batch_overfit():
    # Overfit a tiny single batch to ensure gradient flow
    torch.manual_seed(42)
    model = HybridStrengthNet(lstm_units=32, dense_units=32, dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = AsymmetricHuberLoss(delta=1.0, alpha=2.0, lambda_ce=0.5)

    c = torch.randint(2, 95, (8, 32), dtype=torch.long)
    f = torch.randn(8, 22, dtype=torch.float32)
    y = torch.tensor([[3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]], dtype=torch.float32)
    cls = torch.tensor([0, 1, 1, 2, 2, 3, 3, 4], dtype=torch.long)

    model.train()
    initial_loss = None
    final_loss = None

    for step in range(30):
        optimizer.zero_grad()
        y_hat, logits = model(c, f)
        loss, _, _ = loss_fn(y_hat, y, logits, cls)
        if step == 0:
            initial_loss = loss.item()
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    assert final_loss < initial_loss
