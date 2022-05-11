from typing import Optional

import torch
from torch import nn

from perceiver_io.decoders import BasePerceiverDecoder, TaskDecoder
from perceiver_io.encoder import PerceiverEncoder
from perceiver_io.positional_encoding import PositionalEncoding


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
        sas_param_decoder: TaskDecoder
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
        self.embedding = nn.Linear(encoder.input_dim, encoder.latent_dim)
        torch.nn.init.kaiming_normal_(self.embedding.weight)
        self.pe = PositionalEncoding(encoder.latent_dim)

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
        input_embedding = self.embedding(inputs)
        input_emb_and_pos = self.pe(input_embedding)
        latents = self.encoder(input_emb_and_pos, kv_mask=input_mask)
        clf_outputs = self.sas_model_decoder(
            query=query,
            latents=latents,
            q_mask=query_mask
        )
        reg_outputs = self.sas_param_decoder(
            query=query,
            latents=latents,
            q_mask=query_mask
        )
        return clf_outputs, reg_outputs
