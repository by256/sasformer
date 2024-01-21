import torch
from perceiver_io.attention import CrossAttention
from torch import nn


class PerceiverDecoder(nn.Module):
    def __init__(
        self,
        num_outputs: int,
        num_latents: int,
        latent_dim: int,
        num_heads: int = 1,
        widening_factor: int = 1,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        use_query_residual: bool = False,
    ):
        super().__init__()
        self.q_param = nn.Parameter(torch.randn(num_latents, num_outputs))
        self.cross_attention = CrossAttention(
            q_dim=num_outputs,
            kv_dim=latent_dim,
            num_heads=num_heads,
            widening_factor=widening_factor,
            dropout=dropout,
            attn_dropout=attn_dropout,
            use_query_residual=use_query_residual,
        )

    def forward(self, latents: torch.Tensor):
        batch_size = latents.size(0)
        q_param = self.q_param.repeat(batch_size, 1, 1)
        output = self.cross_attention(q=q_param, kv=latents)
        return output.squeeze(1)
