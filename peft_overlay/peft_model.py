# -*- coding: utf-8 -*-
"""Self-contained MetaPrefix causal LM wrapper for GPT-2/ProtGPT2.

This class implements the Prefix-Tuning interface used by the project without
monkey-patching a separately installed PEFT package. The base language model is
frozen; only the custom prefix encoder is optimized by the meta-learning code.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .config import MetaPrefixConfig
from .mapping import validate_gpt2_compatibility
from .tuners.prefix_tuning import PrefixEncoder


class MetaPrefixForCausalLM(nn.Module):
    """Frozen causal LM with an input-context-conditioned trainable prefix."""

    WEIGHTS_NAME = "prefix_encoder.bin"
    CONFIG_NAME = "metaprefix_config.json"

    def __init__(self, base_model: nn.Module, prefix_config: MetaPrefixConfig):
        super().__init__()
        validate_gpt2_compatibility(base_model)
        self.base_model = base_model
        self.config = base_model.config
        self.prefix_config = prefix_config

        hidden_size = int(getattr(base_model.config, "n_embd", getattr(base_model.config, "hidden_size", -1)))
        num_layers = int(getattr(base_model.config, "n_layer", getattr(base_model.config, "num_hidden_layers", -1)))
        num_heads = int(getattr(base_model.config, "n_head", getattr(base_model.config, "num_attention_heads", -1)))
        expected = (hidden_size, num_layers, num_heads)
        configured = (
            prefix_config.hidden_size,
            prefix_config.num_layers,
            prefix_config.num_attention_heads,
        )
        if expected != configured:
            raise ValueError(
                "Prefix configuration does not match the loaded backbone: "
                f"backbone={expected}, prefix={configured}."
            )

        self.base_model.requires_grad_(False)
        self.prefix_encoder = PrefixEncoder(prefix_config)

    @property
    def device(self) -> torch.device:
        return next(self.prefix_encoder.parameters()).device

    def train(self, mode: bool = True):
        # Keep the frozen backbone in eval mode (dropout disabled) so that the
        # meta-gradient is not contaminated by a changing stochastic backbone.
        super().train(mode)
        self.base_model.eval()
        self.prefix_encoder.train(mode)
        return self

    def _make_past_key_values(
        self,
        condition_input_ids: torch.Tensor,
        condition_attention_mask: torch.Tensor | None = None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        if condition_input_ids.ndim != 2:
            raise ValueError("condition_input_ids must have shape [batch, context_length].")
        if condition_attention_mask is None:
            condition_attention_mask = torch.ones_like(condition_input_ids)
        if condition_attention_mask.shape != condition_input_ids.shape:
            raise ValueError("condition_attention_mask must have the same shape as condition_input_ids.")

        # The backbone is frozen, but gradients must pass through the context
        # embeddings to the trainable cross-attention query and prefix encoder.
        context_embeddings = self.base_model.get_input_embeddings()(condition_input_ids)
        encoded = self.prefix_encoder(
            context_embeddings=context_embeddings,
            context_attention_mask=condition_attention_mask,
        )

        batch_size, prefix_length, _ = encoded.shape
        cfg = self.prefix_config
        # [B, T, L*2*D] -> [L, 2, B, H, T, D/H]
        encoded = encoded.view(
            batch_size,
            prefix_length,
            cfg.num_layers,
            2,
            cfg.num_attention_heads,
            cfg.hidden_size // cfg.num_attention_heads,
        )
        encoded = encoded.permute(2, 3, 0, 4, 1, 5).contiguous()
        return tuple((encoded[layer, 0], encoded[layer, 1]) for layer in range(cfg.num_layers))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        condition_input_ids: torch.Tensor | None = None,
        condition_attention_mask: torch.Tensor | None = None,
        use_cache: bool | None = False,
        **kwargs: Any,
    ):
        """Run a causal LM step with a prefix conditioned only on observed context.

        `input_ids` are decoder-side tokens. `condition_input_ids` are a separate
        observed context (in this project, an EOS sentinel followed by an optional
        N-terminal peptide prefix). The training collator masks the first decoder
        label so the first target residue is predicted from the final context token.
        """
        if input_ids is None or input_ids.ndim != 2:
            raise ValueError("input_ids must be a rank-2 tensor [batch, sequence_length].")
        if condition_input_ids is None:
            raise ValueError(
                "condition_input_ids is required. Pass only observed prefix/context tokens; "
                "do not use the full target sequence as Cross-Attention context."
            )
        if condition_input_ids.shape[0] != input_ids.shape[0]:
            raise ValueError("Decoder and condition batches must have the same batch size.")
        if condition_input_ids.device != input_ids.device:
            raise ValueError("Decoder and condition inputs must be on the same device.")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        elif attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids.")
        if condition_attention_mask is None:
            condition_attention_mask = torch.ones_like(condition_input_ids)
        elif condition_attention_mask.shape != condition_input_ids.shape:
            raise ValueError("condition_attention_mask must match condition_input_ids.")

        past_key_values = self._make_past_key_values(
            condition_input_ids=condition_input_ids,
            condition_attention_mask=condition_attention_mask,
        )
        prefix_mask = torch.ones(
            input_ids.shape[0], self.prefix_config.num_virtual_tokens,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        full_attention_mask = torch.cat((prefix_mask, attention_mask), dim=1)

        return self.base_model(
            input_ids=input_ids,
            attention_mask=full_attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=bool(use_cache),
            **kwargs,
        )

    def save_pretrained(self, save_directory: str | Path, run_config: dict | None = None) -> None:
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        weights = {
            name: tensor.detach().cpu()
            for name, tensor in self.prefix_encoder.state_dict().items()
        }
        torch.save(weights, save_directory / self.WEIGHTS_NAME)
        payload = self.prefix_config.to_dict()
        payload["format_version"] = 1
        payload["base_model_type"] = getattr(self.config, "model_type", "unknown")
        with (save_directory / self.CONFIG_NAME).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        if run_config is not None:
            with (save_directory / "run_config.json").open("w", encoding="utf-8") as handle:
                json.dump(run_config, handle, indent=2, ensure_ascii=False, default=str)

    @classmethod
    def from_pretrained(
        cls,
        base_model: nn.Module,
        adapter_directory: str | Path,
        *,
        trainable: bool = True,
        map_location: str | torch.device = "cpu",
    ) -> "MetaPrefixForCausalLM":
        adapter_directory = Path(adapter_directory)
        config_path = adapter_directory / cls.CONFIG_NAME
        weights_path = adapter_directory / cls.WEIGHTS_NAME
        if not config_path.is_file() or not weights_path.is_file():
            raise FileNotFoundError(
                f"Adapter directory must contain {cls.CONFIG_NAME} and {cls.WEIGHTS_NAME}: "
                f"{adapter_directory}"
            )
        with config_path.open("r", encoding="utf-8") as handle:
            raw_config = json.load(handle)
        prefix_config = MetaPrefixConfig.from_dict(raw_config)
        model = cls(base_model, prefix_config)
        try:
            state = torch.load(weights_path, map_location=map_location, weights_only=True)
        except TypeError:  # compatibility with older PyTorch releases
            state = torch.load(weights_path, map_location=map_location)
        missing, unexpected = model.prefix_encoder.load_state_dict(state, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"Adapter state mismatch. Missing={missing}; unexpected={unexpected}."
            )
        model.prefix_encoder.requires_grad_(bool(trainable))
        model.base_model.requires_grad_(False)
        return model

    @staticmethod
    def _filter_logits(
        logits: torch.Tensor,
        *,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        if top_k > 0:
            k = min(int(top_k), logits.shape[-1])
            threshold = torch.topk(logits, k, dim=-1).values[..., -1, None]
            logits = logits.masked_fill(logits < threshold, float("-inf"))
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            remove = cumulative_probs > float(top_p)
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            filtered = torch.full_like(logits, float("-inf"))
            logits = filtered.scatter(-1, sorted_indices, sorted_logits)
        return logits

    @torch.no_grad()
    def generate(
        self,
        *,
        condition_input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor | None = None,
        max_new_tokens: int = 50,
        min_new_tokens: int = 1,
        temperature: float = 0.9,
        top_p: float = 0.95,
        top_k: int = 50,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """Autoregressive generation using the same context-conditioned prefix path."""
        self.eval()
        if condition_input_ids.ndim != 2 or condition_input_ids.shape[1] < 1:
            raise ValueError("condition_input_ids must contain at least one context token.")
        if decoder_input_ids is None:
            # The full observed context also enters the causal decoder stream,
            # preserving its order and positional encoding.
            decoder_input_ids = condition_input_ids
        if decoder_input_ids.ndim != 2 or decoder_input_ids.shape[0] != condition_input_ids.shape[0]:
            raise ValueError("decoder_input_ids must have shape [batch, decoder_length].")
        if decoder_input_ids.shape[1] < 1:
            raise ValueError("decoder_input_ids must contain at least one token.")
        if decoder_input_ids.device != condition_input_ids.device:
            raise ValueError("Decoder and condition inputs must be on the same device.")
        if temperature <= 0:
            raise ValueError("temperature must be > 0.")
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1].")
        if max_new_tokens < 1:
            return decoder_input_ids

        batch_size = decoder_input_ids.shape[0]
        if eos_token_id is None:
            eos_token_id = getattr(self.config, "eos_token_id", None)
        if pad_token_id is None:
            pad_token_id = eos_token_id
        if eos_token_id is None or pad_token_id is None:
            raise ValueError("Both eos_token_id and pad_token_id must be available.")

        condition_mask = torch.ones_like(condition_input_ids)
        past_key_values = self._make_past_key_values(condition_input_ids, condition_mask)
        prefix_mask = torch.ones(
            batch_size, self.prefix_config.num_virtual_tokens,
            dtype=torch.long, device=decoder_input_ids.device,
        )
        decoder_mask = torch.ones_like(decoder_input_ids)
        full_attention_mask = torch.cat((prefix_mask, decoder_mask), dim=1)

        outputs = self.base_model(
            input_ids=decoder_input_ids,
            attention_mask=full_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        cache = outputs.past_key_values
        if cache is None:
            raise RuntimeError("The backbone did not return past_key_values; generation requires a KV cache.")
        next_logits = outputs.logits[:, -1, :]
        sequences = decoder_input_ids
        finished = torch.zeros(batch_size, dtype=torch.bool, device=decoder_input_ids.device)

        for step in range(max_new_tokens):
            logits = next_logits / float(temperature)
            if step < min_new_tokens:
                logits[:, int(eos_token_id)] = float("-inf")
            if do_sample:
                logits = self._filter_logits(logits, top_k=top_k, top_p=top_p)
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_token = torch.argmax(logits, dim=-1)

            next_token = torch.where(
                finished,
                torch.full_like(next_token, int(pad_token_id)),
                next_token,
            )
            sequences = torch.cat((sequences, next_token[:, None]), dim=1)
            newly_finished = next_token.eq(int(eos_token_id))
            finished = finished | newly_finished
            if bool(finished.all()):
                break

            # A row that already emitted EOS receives a masked padding token on
            # subsequent decoding iterations; other rows continue normally.
            step_mask = (~finished).to(dtype=full_attention_mask.dtype).unsqueeze(1)
            full_attention_mask = torch.cat((full_attention_mask, step_mask), dim=1)
            outputs = self.base_model(
                input_ids=next_token[:, None],
                attention_mask=full_attention_mask,
                past_key_values=cache,
                use_cache=True,
            )
            cache = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :]

        return sequences
