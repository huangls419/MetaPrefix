# -*- coding: utf-8 -*-
"""MetaPrefix's dynamic PrefixEncoder: four-layer MLP + multi-head cross-attention."""
from __future__ import annotations

import torch
from torch import nn

from ..config import MetaPrefixConfig


class PrefixEncoder(nn.Module):
    """Encode observed N-terminal context into per-layer prefix keys and values.

    The virtual prompt tokens are transformed by a four-Linear-layer MLP and then
    act as Cross-Attention queries. The *observed context* embeddings are keys and
    values. In autoregressive training, context tokens are restricted to residues
    that precede the prediction target; the encoder never attends to future target
    residues.
    """

    def __init__(self, config: MetaPrefixConfig):
        super().__init__()
        self.config = config
        d = config.hidden_size
        h = config.encoder_hidden_size
        self.embedding = nn.Embedding(config.num_virtual_tokens, d)
        self.transform = nn.Sequential(
            nn.Linear(d, h),
            nn.Tanh(),
            nn.Linear(h, h),
            nn.Tanh(),
            nn.Linear(h, h),
            nn.Tanh(),
            nn.Linear(h, d),
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            batch_first=True,
        )
        self.output_transform = nn.Linear(
            d, config.num_layers * 2 * d
        )
        self.register_buffer(
            "virtual_token_ids",
            torch.arange(config.num_virtual_tokens, dtype=torch.long),
            persistent=False,
        )

    def forward(
        self,
        context_embeddings: torch.Tensor,
        context_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if context_embeddings.ndim != 3:
            raise ValueError(
                "context_embeddings must have shape [batch, context_length, hidden_size]; "
                f"received {tuple(context_embeddings.shape)}."
            )
        batch_size, context_length, hidden_size = context_embeddings.shape
        if context_length == 0:
            raise ValueError("At least one context token is required (use EOS as the empty-context token).")
        if hidden_size != self.config.hidden_size:
            raise ValueError(
                f"Context hidden size {hidden_size} does not match configured size {self.config.hidden_size}."
            )
        if context_attention_mask is None:
            context_attention_mask = torch.ones(
                batch_size, context_length,
                dtype=torch.bool,
                device=context_embeddings.device,
            )
        else:
            if tuple(context_attention_mask.shape) != (batch_size, context_length):
                raise ValueError(
                    "context_attention_mask must match the first two context dimensions."
                )
            context_attention_mask = context_attention_mask.to(
                device=context_embeddings.device, dtype=torch.bool
            )
        if torch.any(context_attention_mask.sum(dim=1) == 0):
            raise ValueError("Every context row must contain at least one unmasked token.")

        virtual_ids = self.virtual_token_ids.unsqueeze(0).expand(batch_size, -1)
        query = self.transform(self.embedding(virtual_ids))
        attn_output, _ = self.cross_attention(
            query=query,
            key=context_embeddings,
            value=context_embeddings,
            key_padding_mask=~context_attention_mask,
            # Request attention weights (discarded below) to select the explicit
            # matmul/softmax path in PyTorch MultiheadAttention. Some fused SDPA
            # backends do not implement double backward, which Stage 1 MAML needs.
            need_weights=True,
        )
        return self.output_transform(attn_output)
