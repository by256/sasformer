from typing import Optional

import torch
from torch import nn


class MLP(nn.Module):
    """Transformer MLP."""

    def __init__(self, embed_dim: int, widening_factor: int = 4, dropout: float = 0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * widening_factor),
            nn.GELU(),
            nn.Linear(embed_dim * widening_factor, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        return self.mlp(x)


class SelfAttention(nn.Module):
    """Perceiver IO self-attention module."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        widening_factor: int = 4,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.input_layer_norm = nn.LayerNorm(embed_dim)
        self.qkv_layer_norm = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=attn_dropout, batch_first=True
        )
        self.mlp = MLP(embed_dim, widening_factor, dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        x_norm = self.input_layer_norm(x)
        x_qkv = self.attention(x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask, attn_mask=attn_mask)[0]
        x_qkv = x_qkv + x
        return x_qkv + self.mlp(self.qkv_layer_norm(x_qkv))


class CrossAttention(nn.Module):
    """Perceiver IO cross-attention module."""

    def __init__(
        self,
        q_dim: int,
        kv_dim: int,
        num_heads: int,
        widening_factor: int = 4,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        use_query_residual: bool = False,
    ):
        super().__init__()
        self.use_query_residual = use_query_residual
        self.q_layer_norm = nn.LayerNorm(q_dim)
        self.kv_layer_norm = nn.LayerNorm(kv_dim)
        self.qkv_layer_norm = nn.LayerNorm(q_dim)
        self.attention = nn.MultiheadAttention(
            q_dim, num_heads=num_heads, dropout=attn_dropout, kdim=kv_dim, vdim=kv_dim, batch_first=True
        )
        self.mlp = MLP(q_dim, widening_factor, dropout)

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        q_norm = self.q_layer_norm(q)
        kv_norm = self.kv_layer_norm(kv)
        x_qkv = self.attention(q_norm, kv_norm, kv_norm, key_padding_mask=key_padding_mask, attn_mask=attn_mask)[0]
        if self.use_query_residual:
            x_qkv = x_qkv + q
        return x_qkv + self.mlp(self.qkv_layer_norm(x_qkv))
