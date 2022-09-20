from typing import Optional

import torch
from torch import nn

from perceiver_io.decoders import BasePerceiverDecoder, TaskDecoder
from perceiver_io.encoder import PerceiverEncoder
from perceiver_io.positional_encoding import PositionalEncoding


class TokenScaleAndPositionEmbedding(nn.Module):
    def __init__(self, latent_dim, n_bins=256, seq_len=256):
        super().__init__()
        self.token_embedding = nn.Embedding(
            num_embeddings=n_bins, embedding_dim=latent_dim)
        self.scale_embedding = nn.Embedding(
            num_embeddings=n_bins, embedding_dim=latent_dim)
        self.pos_embedding = nn.Embedding(
            num_embeddings=seq_len, embedding_dim=latent_dim)
        self.pos_idxs = torch.arange(seq_len).unsqueeze(0)

    def forward(self, x):
        if self.pos_idxs.device != x.device:
            self.pos_idxs = self.pos_idxs.to(x.device)
        # TODO: remove token squeeze after updating data
        token = self.token_embedding(x[:, :-1, :]).squeeze()
        scale = self.scale_embedding(x[:, -1, :])
        pos = self.pos_embedding(self.pos_idxs.repeat(x.shape[0], 1))
        return token + scale + pos


class PerceiverIO(nn.Module):
    """Perceiver IO encoder-decoder architecture."""

    def __init__(
        self,
        encoder: PerceiverEncoder,
        decoder: BasePerceiverDecoder
    ):
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
        outputs = self.decoder(
            query=query,
            latents=latents,
            q_mask=query_mask
        )
        return outputs


class SASPerceiverIO(nn.Module):
    """SAS Perceiver IO encoder-decoder architecture."""

    def __init__(
        self,
        encoder: PerceiverEncoder,
        sas_model_decoder: TaskDecoder,
        sas_param_decoder: TaskDecoder,
        n_bins: int = 256,
        seq_len: int = 256
    ):
        """Constructor.

        Args:
            encoder: Instance of Perceiver IO encoder.
            sas_model_decoder: Instance of TaskDecoder.
            sas_param_decoder: Instance of TaskDecoder.
        """
        super().__init__()
        self.encoder = encoder
        self.sas_model_decoder = sas_model_decoder
        self.sas_param_decoder = sas_param_decoder
        self.embedding = TokenScaleAndPositionEmbedding(
            encoder.latent_dim, n_bins, seq_len)

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
        emb = self.embedding(inputs)
        latents = self.encoder(emb, kv_mask=input_mask)
        clf_outputs = self.sas_model_decoder(latents)
        reg_outputs = self.sas_param_decoder(latents)
        return clf_outputs, reg_outputs
