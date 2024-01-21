from typing import Optional

import torch
from perceiver_io.attention import CrossAttention, SelfAttention
from torch import nn


class PerceiverEncoder(nn.Module):
    """Perceiver encoder module. Consists of two components: cross-attention
    module that maps an input tensor and a trainable latent tensor to a latent
    tensor and a stacked Transformer blocks with shared weights.
    """

    def __init__(
        self,
        num_latents: int,
        latent_dim: int,
        num_blocks: int = 4,
        num_self_attn_per_block: int = 2,
        num_cross_attn_heads: int = 1,
        num_self_attn_heads: int = 1,
        cross_attn_widening_factor: int = 1,
        self_attn_widening_factor: int = 1,
        dropout: float = 0.0,
        cross_attn_dropout: float = 0.0,
        self_attn_dropout: float = 0.0,
        use_query_residual: bool = False,
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
            cross_attn_dropout: Cross-attention scores dropout probability.
                Defaults to 0.
            self_attn_dropout: Self-attention scores dropout probability.
                Defaults to 0.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.latent_param = nn.Parameter(torch.randn(num_latents, latent_dim))

        self.cross_attn = CrossAttention(
            q_dim=latent_dim,
            kv_dim=latent_dim,
            num_heads=num_cross_attn_heads,
            widening_factor=cross_attn_widening_factor,
            dropout=dropout,
            attn_dropout=cross_attn_dropout,
            use_query_residual=use_query_residual,
        )
        block = self.self_attention_block(
            num_self_attn_per_block,
            latent_dim,
            widening_factor=self_attn_widening_factor,
            num_heads=num_self_attn_heads,
            dropout=dropout,
            attn_dropout=self_attn_dropout,
        )
        self.self_attention_blocks = nn.Sequential(*[block for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: Input tensor of shape (B, M, C).
            key_padding_mask: Same as key_padding_mask in torch.nn.MultiheadAttention. Defaults to None.

        Returns:
            Latent tensor.
        """
        batch_size = x.size(0)
        latent_param = self.latent_param.repeat(batch_size, 1, 1)
        latents = self.cross_attn(q=latent_param, kv=x, key_padding_mask=key_padding_mask)
        return self.self_attention_blocks(latents)

    def self_attention_block(
        self,
        num_self_attn_per_block: int,
        embed_dim: int,
        num_heads: int = 1,
        widening_factor: int = 4,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ):
        return nn.Sequential(
            *[
                SelfAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    widening_factor=widening_factor,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                )
                for _ in range(num_self_attn_per_block)
            ]
        )
