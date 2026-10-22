"""HybridStrengthNet: Hybrid Neural and Pattern-Aware Guessability Model (PyTorch)."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class AdditiveAttentionPooling(nn.Module):
    """Additive attention pooling over sequence with padding mask."""

    def __init__(self, hidden_dim: int, attn_dim: int = 128):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Args:

        h: [B, T, hidden_dim]
        mask: [B, T] (1 for valid tokens, 0 for pad)
        Returns:
            pooled: [B, hidden_dim]
        """
        # score = v(tanh(W h)) -> [B, T, 1]
        score = self.v(torch.tanh(self.proj(h))).squeeze(-1)  # [B, T]
        # Mask padding
        score = score.masked_fill(~mask.bool(), -1e9)
        alpha = F.softmax(score, dim=-1)  # [B, T]
        alpha = torch.where(torch.isnan(alpha), torch.zeros_like(alpha), alpha)
        # alpha unsqueeze -> [B, 1, T] @ [B, T, hidden_dim] -> [B, 1, hidden_dim]
        pooled = torch.bmm(alpha.unsqueeze(1), h).squeeze(1)  # [B, hidden_dim]
        return pooled


class HybridStrengthNet(nn.Module):
    """Hybrid Neural and Pattern-Aware Guessability Model.

    Inputs:
        char_ids: [B, 32] (int64)
        features: [B, F] (float32)
    Outputs:
        y_hat: [B, 1] (float32)
        class_logits: [B, 5] (float32)
    """

    def __init__(
        self,
        vocab_size: int = 98,
        embed_dim: int = 32,
        lstm_layers: int = 2,
        lstm_units: int = 128,
        feature_dim: int = 22,
        feature_mlp: int = 64,
        dense_units: int = 128,
        dropout: float = 0.3,
        use_features: bool = True,
        use_attention: bool = True,
    ):
        super().__init__()
        self.use_features = use_features
        self.use_attention = use_attention
        self.feature_dim = feature_dim

        # Sequence branch
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_units,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        seq_hidden_dim = lstm_units * 2  # 256

        if use_attention:
            self.attn_pool = AdditiveAttentionPooling(seq_hidden_dim, attn_dim=128)

        # Feature branch
        if use_features:
            self.feat_fc = nn.Linear(feature_dim, feature_mlp)
            self.feat_dropout = nn.Dropout(dropout)
            fusion_in = seq_hidden_dim + feature_mlp
        else:
            fusion_in = seq_hidden_dim

        # Fusion layer
        self.fusion_fc = nn.Linear(fusion_in, dense_units)
        self.fusion_dropout = nn.Dropout(dropout)

        # Output heads
        self.reg_head = nn.Linear(dense_units, 1)
        self.cls_head = nn.Linear(dense_units, 5)

    def forward(
        self, char_ids: torch.Tensor, features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = (char_ids != 0).long()
        x_emb = self.embedding(char_ids)  # [B, T, embed_dim]
        h_seq, _ = self.bilstm(x_emb)      # [B, T, 256]

        if self.use_attention:
            h_pooled = self.attn_pool(h_seq, mask)  # [B, 256]
        else:
            # Fallback: masked mean pooling
            mask_float = mask.unsqueeze(-1).float()
            sum_h = torch.sum(h_seq * mask_float, dim=1)
            lengths = torch.clamp(torch.sum(mask_float, dim=1), min=1.0)
            h_pooled = sum_h / lengths

        if self.use_features:
            if features is None:
                # Dummy features if not supplied
                features = torch.zeros(char_ids.shape[0], self.feature_dim, device=char_ids.device)
            h_f = F.relu(self.feat_fc(features))
            h_f = self.feat_dropout(h_f)
            h_combined = torch.cat([h_pooled, h_f], dim=-1)
        else:
            h_combined = h_pooled

        h_dense = F.relu(self.fusion_fc(h_combined))
        h_dense = self.fusion_dropout(h_dense)

        y_hat = self.reg_head(h_dense)
        class_logits = self.cls_head(h_dense)

        return y_hat, class_logits


class AsymmetricHuberLoss(nn.Module):
    """Asymmetric Huber Loss penalizing overestimation (e > 0) with weight alpha."""

    def __init__(self, delta: float = 1.0, alpha: float = 2.0, lambda_ce: float = 0.5):
        super().__init__()
        self.delta = delta
        self.alpha = alpha
        self.lambda_ce = lambda_ce

    def forward(
        self,
        y_hat: torch.Tensor,
        y_true: torch.Tensor,
        class_logits: Optional[torch.Tensor] = None,
        cls_true: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        e = y_hat.squeeze(-1) - y_true.squeeze(-1)
        abs_e = torch.abs(e)

        # Standard Huber
        linear_mask = abs_e > self.delta
        huber = torch.where(
            linear_mask,
            self.delta * (abs_e - 0.5 * self.delta),
            0.5 * (e ** 2),
        )

        # Asymmetric weights: alpha if e > 0 (overestimate), 1.0 otherwise
        weights = torch.where(e > 0, self.alpha, 1.0)
        l_reg = torch.mean(weights * huber)

        l_ce = torch.tensor(0.0, device=y_hat.device)
        if class_logits is not None and cls_true is not None:
            l_ce = F.cross_entropy(class_logits, cls_true.long())

        total_loss = l_reg + self.lambda_ce * l_ce
        return total_loss, l_reg, l_ce
