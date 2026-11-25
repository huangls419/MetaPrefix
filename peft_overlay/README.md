# Active MetaPrefix implementation

The Python package in this folder is the active, self-contained implementation used by the training scripts. It does not patch or replace the installed Hugging Face PEFT package.

- `config.py`: stable configuration saved with every adapter.
- `tuners/prefix_tuning.py`: four-Linear-layer MLP, multi-head Cross-Attention, and projection to per-layer key/value tensors.
- `peft_model.py`: frozen GPT-2/ProtGPT2 backbone wrapper, adapter save/load, and autoregressive generation.
- `mapping.py`: GPT-style architecture dimension inference and checks.

The uploaded files from the previous PEFT source-overlay approach are preserved in `reference/legacy_peft_overlay/` at the project root. They are not imported by the active model. This avoids the previous requirement to copy files into `site-packages/peft`, the missing `tools.patch_peft` bootstrap, and the incompatible inference-mode PrefixEncoder state layout.

## Context interface

Training and inference must supply `condition_input_ids` separately from decoder `input_ids`. This is deliberate: Cross-Attention must see only observed context tokens, not the complete target sequence being predicted. The collator emits this context automatically. Direct calls to `MetaPrefixForCausalLM.forward()` without `condition_input_ids` raise an error to prevent accidental future-token leakage.

The Cross-Attention key padding mask is derived from `condition_attention_mask`, so padded context positions do not contribute to the dynamic prefix.

## Input/output dimensions

Let `D` be ProtGPT2 hidden size, `L` its Transformer layer count, `H` its attention-head count, and `P` the virtual-prefix length. The encoder returns `[batch, P, 2*L*D]`. This is reshaped into `L` key/value pairs, each with shape `[batch, H, P, D/H]`, and passed to the GPT-2-compatible `past_key_values` input.
