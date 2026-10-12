# peft_patch/prefix_tuning.py
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from ..utils import PeftType, PromptLearningConfig


@dataclass
class PrefixTuningConfig(PromptLearningConfig):
    encoder_hidden_size: int = field(
        default=None,
        metadata={"help": "The hidden size of the encoder"},
    )
    prefix_projection: bool = field(
        default=False,
        metadata={"help": "Whether to project the prefix tokens"},
    )

    def __post_init__(self):
        self.peft_type = PeftType.PREFIX_TUNING


class PrefixEncoder(torch.nn.Module):
    """
    MetaPrefix Modified PrefixEncoder:
    1. Four-layer MLP projection replaces the original two-layer MLP;
    2. Added Multi-Head Cross-Attention to let the prefix interact with the input sequence hidden states;
    3. The output is still in the past_key_values shape required by PEFT Prefix-Tuning.
    """

    def __init__(self, config):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        self.token_dim = config.token_dim
        self.num_layers = config.num_layers
        self.encoder_hidden_size = config.encoder_hidden_size
        self.num_virtual_tokens = config.num_virtual_tokens
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.token_dim // self.num_attention_heads

        if self.prefix_projection and not getattr(config, "inference_mode", False):
            self.embedding = nn.Embedding(self.num_virtual_tokens, self.token_dim)

            #4-layer MLP: input layer -> two hidden layers -> output layer
            self.transform = nn.Sequential(
                nn.Linear(self.token_dim, self.encoder_hidden_size),
                nn.Tanh(),
                nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size),
                nn.Tanh(),
                nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size),
                nn.Tanh(),
                nn.Linear(self.encoder_hidden_size, self.token_dim),
            )

            self.cross_attention = nn.MultiheadAttention(
                embed_dim=self.token_dim,
                num_heads=self.num_attention_heads,
                batch_first=False,
            )

            self.output_transform = nn.Linear(
                self.token_dim,
                self.num_layers * 2 * self.token_dim,
            )
        else:
            self.embedding = nn.Embedding(
                self.num_virtual_tokens,
                self.num_layers * 2 * self.token_dim,
            )

    def forward(self, prefix: torch.Tensor, hidden_states: torch.Tensor = None):
        if not self.prefix_projection:
            return self.embedding(prefix)

        # prefix_tokens: (B, T, D)
        prefix_tokens = self.embedding(prefix)
        prefix_tokens = self.transform(prefix_tokens)

        if hidden_states is None:
            hidden_states = prefix_tokens

        query = prefix_tokens.permute(1, 0, 2)        # (T, B, D)
        key_value = hidden_states.permute(1, 0, 2)    # (S, B, D)

        attn_output, _ = self.cross_attention(
            query=query,
            key=key_value,
            value=key_value,
        )
        attn_output = attn_output.permute(1, 0, 2)    # (B, T, D)
        attn_output = self.output_transform(attn_output)
        return attn_output