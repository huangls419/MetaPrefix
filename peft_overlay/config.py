# -*- coding: utf-8 -*-
"""Configuration for the self-contained MetaPrefix prefix-tuning adapter."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class MetaPrefixConfig:
    num_virtual_tokens: int = 10
    encoder_hidden_size: int = 1280
    hidden_size: int = 1280
    num_layers: int = 36
    num_attention_heads: int = 20
    mlp_num_linear_layers: int = 4
    attention_dropout: float = 0.0

    def __post_init__(self) -> None:
        values = {
            "num_virtual_tokens": self.num_virtual_tokens,
            "encoder_hidden_size": self.encoder_hidden_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "mlp_num_linear_layers": self.mlp_num_linear_layers,
        }
        for name, value in values.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer; got {value!r}.")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads "
                f"({self.hidden_size} % {self.num_attention_heads} != 0)."
            )
        if self.mlp_num_linear_layers != 4:
            raise ValueError("This implementation follows the four-Linear-layer MetaPrefix encoder.")
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError("attention_dropout must be in [0, 1).")

    @classmethod
    def from_transformer_config(
        cls,
        transformer_config: Any,
        *,
        num_virtual_tokens: int = 10,
        encoder_hidden_size: int | None = None,
        attention_dropout: float = 0.0,
    ) -> "MetaPrefixConfig":
        def first_attr(*names: str) -> int:
            for name in names:
                value = getattr(transformer_config, name, None)
                if value is not None:
                    return int(value)
            raise ValueError(
                "Cannot infer GPT-style model dimensions from config. "
                f"Tried attributes: {', '.join(names)}."
            )

        hidden_size = first_attr("n_embd", "hidden_size", "d_model")
        num_layers = first_attr("n_layer", "num_hidden_layers", "num_layers")
        num_heads = first_attr("n_head", "num_attention_heads", "num_heads")
        return cls(
            num_virtual_tokens=int(num_virtual_tokens),
            encoder_hidden_size=int(encoder_hidden_size or hidden_size),
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_attention_heads=num_heads,
            attention_dropout=float(attention_dropout),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "MetaPrefixConfig":
        allowed = set(cls.__dataclass_fields__)
        clean = {k: v for k, v in values.items() if k in allowed}
        return cls(**clean)
