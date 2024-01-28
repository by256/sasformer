from typing import Optional

import torch
from torch import nn

from perceiver_io.attention import CrossAttention, SelfAttention


class PerceiverEncoder(nn.Module):
    """Perceiver encoder module. Consists of two components: cross-attention
    module that maps an input tensor and a trainable latent tensor to a latent
    tensor and a stacked Transformer blocks with shared weights.
    """

    def __init__(
        self,
        num_latents: int,
        latent_dim: int,
        num_self_attn_per_block: int = 2,
        num_blocks: int = 4,
        qk_out_dim: Optional[int] = None,
        v_out_dim: Optional[int] = None,
        num_cross_attn_heads: int = 1,
        num_self_attn_heads: int = 8,
        cross_attn_widening_factor: int = 1,
        self_attn_widening_factor: int = 1,
        use_query_residual: bool = False,
        dropout: float = 0.0,
        cross_attention_dropout: float = 0.0,
        self_attention_dropout: float = 0.0
    ):
        """Constructor.

        Args:
            num_latents: Number of latent vectors.
            latent_dim: Dimension of latent vector.
            num_self_attn_per_block: Number of self-attention modules per
                transformer block. Defaults to 2.
            num_blocks: Number of transformer blocks. Defaults to 4.
            qk_out_dim: Size of Query and Key matrices last dimension.
                Defaults to None.
            v_out_dim: Size of Value matrix last dimension.
                Defaults to None.
            num_cross_attn_heads: Number of cross-attention heads.
                Defaults to 1.
            num_self_attn_heads: Number of self-attention heads.
                Defaults to 8.
            cross_attn_widening_factor: Widening factor in cross-attention
                feed-forward layer. Defaults to 1.
            self_attn_widening_factor: Widening factor in self-attention
                feed-forward layer. Defaults to 1.
            use_query_residual: Indicates whether to use query residual in
                cross-attention. Defaults to True.
            dropout: Feed-forward dropout probability. Defaults to 0.
            cross_attention_dropout: Cross-attention scores dropout probability.
                Defaults to 0.
            self_attention_dropout: Self-attention scores dropout probability.
                Defaults to 0.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        self.cross_attn = CrossAttention(
            kv_dim=latent_dim,
            q_dim=latent_dim,
            widening_factor=cross_attn_widening_factor,
            num_heads=num_cross_attn_heads,
            qk_out_dim=qk_out_dim,
            v_out_dim=v_out_dim,
            use_query_residual=use_query_residual,
            dropout=dropout,
            attention_dropout=cross_attention_dropout
        )
        block = self.self_attention_block(
            num_self_attn_per_block,
            latent_dim,
            widening_factor=self_attn_widening_factor,
            num_heads=num_self_attn_heads,
            qk_out_dim=qk_out_dim,
            v_out_dim=v_out_dim,
            dropout=dropout,
            attention_dropout=self_attention_dropout
        )
        self.self_attention_blocks = nn.Sequential(
            *[block for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor, kv_mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: Input tensor of shape (B, M, C).
            kv_mask: Input mask tensor of shape (B, M). Mask values selected
                in [0, 1]. Defaults to None.

        Returns:
            Latent tensor.
        """
        batch_size = x.size(0)
        if kv_mask is not None:
            kv_mask = kv_mask[:, None, None, :]

        latents = self.cross_attn(
            inputs_kv=x,
            inputs_q=self.latents.repeat(batch_size, 1, 1),
            attention_mask=kv_mask
        )
        return self.self_attention_blocks(latents)

    def self_attention_block(
            self,
            num_self_attn_per_block: int,
            hidden_dim: int,
            qk_out_dim: Optional[int] = None,
            v_out_dim: Optional[int] = None,
            widening_factor: int = 4,
            num_heads: int = 1,
            dropout: float = 0.0,
            attention_dropout: float = 0.0):

        return nn.Sequential(*[
            SelfAttention(
                hidden_dim=hidden_dim,
                widening_factor=widening_factor,
                num_heads=num_heads,
                qk_out_dim=qk_out_dim,
                v_out_dim=v_out_dim,
                dropout=dropout,
                attention_dropout=attention_dropout
            ) for _ in range(num_self_attn_per_block)
        ])
