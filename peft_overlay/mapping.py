# -*- coding: utf-8 -*-
"""Small architecture mapping used by the self-contained MetaPrefix wrapper."""
from __future__ import annotations

from typing import Any


def infer_gpt_dimensions(config: Any) -> tuple[int, int, int]:
    """Return (hidden_size, num_layers, num_attention_heads) for GPT-like models."""
    def get_first(*names: str) -> int:
        for name in names:
            value = getattr(config, name, None)
            if value is not None:
                return int(value)
        raise ValueError(f"Missing transformer dimension; checked: {', '.join(names)}")

    hidden = get_first("n_embd", "hidden_size", "d_model")
    layers = get_first("n_layer", "num_hidden_layers", "num_layers")
    heads = get_first("n_head", "num_attention_heads", "num_heads")
    if hidden % heads:
        raise ValueError(f"Hidden size {hidden} is not divisible by attention heads {heads}.")
    return hidden, layers, heads


def validate_gpt2_compatibility(model: Any) -> None:
    """Fail early unless the base model exposes GPT-2-style token embeddings and KV cache."""
    if not hasattr(model, "get_input_embeddings") or model.get_input_embeddings() is None:
        raise TypeError("Base LM must expose get_input_embeddings().")
    if not hasattr(model, "config"):
        raise TypeError("Base LM must expose a transformer config.")
    infer_gpt_dimensions(model.config)
    if not callable(getattr(model, "forward", None)):
        raise TypeError("Base LM must expose forward().")
