from typing import Optional

import numpy as np
import torch
from perceiver_io.decoders import PerceiverDecoder
from perceiver_io.encoder import PerceiverEncoder
from torch import nn


def sinusoids(length, channels, max_timescale=512):
    """Returns sinusoids for positional embedding.
    https://github.com/openai/whisper - Jong Wook Kim 2022 (OpenAI)"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, latent_dim, n_bins=256, seq_len=256):
        super().__init__()
        self.token_embedding = nn.Embedding(num_embeddings=n_bins, embedding_dim=latent_dim)
        self.pos_embedding = sinusoids(seq_len, latent_dim).unsqueeze(0)

    def forward(self, x):
        if self.pos_embedding.device != x.device:
            self.pos_embedding = self.pos_embedding.to(x.device)
        # TODO: remove token squeeze after updating data
        token = self.token_embedding(x).squeeze()
        pos = self.pos_embedding
        return token + pos


class PerceiverIO(nn.Module):
    """Perceiver IO encoder-decoder architecture."""

    def __init__(self, encoder: PerceiverEncoder, decoder: PerceiverDecoder):
        """Constructor.

        Args:
            encoder: Instance of Perceiver IO encoder.
            decoder: Instance of Perceiver IO decoder.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        inputs: torch.Tensor,
        query: Optional[torch.Tensor] = None,
        input_mask: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            inputs: Input tensor.
            query: Decoder query tensor. Can be a trainable or hand-made.
                Defaults to None.
            input_mask: Input mask tensor. Mask values selected in [0, 1].
                Defaults to None.
            query_mask: Decoder query mask tensor. Mask values selected in
                [0, 1]. Defaults to None.

        Returns:
            Output tensor.
        """
        latents = self.encoder(inputs, kv_mask=input_mask)
        outputs = self.decoder(query=query, latents=latents, q_mask=query_mask)
        return outputs


class SASPerceiverIO(nn.Module):
    """SAS Perceiver IO encoder-decoder architecture."""

    def __init__(
        self,
        encoder: PerceiverEncoder,
        sas_model_decoder: PerceiverDecoder,
        sas_param_decoder: PerceiverDecoder,
        n_bins: int = 256,
        seq_len: int = 511,
    ):
        """Constructor.

        Args:
            encoder: Instance of PerceiverEncoder.
            sas_model_decoder: Instance of PerceiverDecoder.
            sas_param_decoder: Instance of PerceiverDecoder.
        """
        super().__init__()
        self.encoder = encoder
        self.sas_model_decoder = sas_model_decoder
        self.sas_param_decoder = sas_param_decoder
        self.embedding = TokenAndPositionEmbedding(encoder.latent_dim, n_bins, seq_len)

    def forward(self, inputs: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        """
        Args:
            inputs: Input tensor.
            query: Decoder query tensor. Can be a trainable or hand-made.
                Defaults to None.
            input_mask: Input mask tensor. Mask values selected in [0, 1].
                Defaults to None.
            query_mask: Decoder query mask tensor. Mask values selected in
                [0, 1]. Defaults to None.

        Returns:
            Output tensor.
        """
        emb = self.embedding(inputs)
        latents = self.encoder(emb, key_padding_mask)
        clf_outputs = self.sas_model_decoder(latents)
        reg_outputs = self.sas_param_decoder(latents)
        return clf_outputs, reg_outputs
