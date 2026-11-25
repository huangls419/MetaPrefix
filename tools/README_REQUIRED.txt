REQUIRED ORIGINAL MODEL PATCH

The uploaded Stage 1 and Stage 2 scripts both import `tools.patch_peft` before building
or loading the PEFT adapter. The actual `tools/patch_peft.py` source file was not among
the files supplied for this reconstruction. The manuscript describes an enhanced prefix
encoder with MLP and cross-attention, but those details alone are not enough to recreate
the exact implementation and state-dict key layout.

Before training, copy the EXACT `tools/patch_peft.py` from the original MetaPrefix code
repository into this directory. Do not replace it with a guessed implementation.

The code in `scripts/common.py` intentionally stops with a clear error when this exact
patch cannot be imported. That protects the reproduction from silently changing the
model architecture.
