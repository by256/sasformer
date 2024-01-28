import math
from typing import Sequence

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000, scale: float = 1.0):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term) * scale
        pe[0, :, 1::2] = torch.cos(position * div_term) * scale
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        batch_size = x.size(0)
        x = torch.cat(
            [x, self.pe[:, :x.size(1), :].repeat(batch_size, 1, 1)], dim=-1)
        return self.dropout(x)


def fourier_encoding(
    dims: Sequence[int],
    num_bands: int,
    resolutions: Sequence[int],
    concatenate_positions: bool = True
) -> torch.Tensor:
    """Generate Fourier positional encodings.

    Args:
        dims: Sequence of dimensions.
        num_bands: Number of frequency bands.
        resolutions: Sequence of resolutions for each dimension.
        concatenate_positions: Indicates whether to concatenate positions to
            the encodings. Defaults to True.

    Returns:
        Tensor of shape (dims[0], ..., dims[d], num_bands * D)
        where D is number of dimensions.
    """
    # make sure that number of resolutions is equals to number of dimensions
    assert len(resolutions) == len(dims)

    # generate a position indices grid of shape (dims[0], ..., dims[d], D)
    ranges = [torch.linspace(-1, 1, dim) for dim in dims]
    grid = torch.meshgrid(*ranges)
    grid = torch.stack(grid, dim=-1)

    # frequency bands for each resolution of shape (len(resolutions), num_bands)
    freq_bands = torch.stack([
        torch.linspace(1, res / 2, steps=num_bands)
        for res in resolutions
    ], dim=0)

    # frequency features of shape (dims[1], ..., dims[d], D, num_bands)
    features = grid[..., None] * freq_bands[None, ...]
    sin = torch.sin(features * math.pi)
    cos = torch.cos(features * math.pi)
    features = torch.cat([sin, cos], dim=-1)

    # reshape the encodings as a tensor of shape
    # (dims[0], dims[1], ..., dims[d], num_bands * D)
    features = features.view(*grid.shape[:-1], -1)

    if concatenate_positions:
        features = torch.cat([features, grid], dim=-1)
    return features
