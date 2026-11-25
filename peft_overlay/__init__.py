# -*- coding: utf-8 -*-
"""Self-contained MetaPrefix implementation (no monkey-patching of PEFT required)."""
from .config import MetaPrefixConfig
from .peft_model import MetaPrefixForCausalLM
from .tuners.prefix_tuning import PrefixEncoder

__all__ = ["MetaPrefixConfig", "MetaPrefixForCausalLM", "PrefixEncoder"]
