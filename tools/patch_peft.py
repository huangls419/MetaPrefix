"""Patch PEFT to use MetaPrefix's custom PrefixEncoder/PrefixTuningConfig.

Import this module before constructing any PEFT model (e.g. before get_peft_model
or PeftModel.from_pretrained is called).
"""

import importlib
import sys

import peft

from peft_patch.prefix_tuning import (
    PrefixEncoder as MetaPrefixEncoder,
    PrefixTuningConfig as MetaPrefixTuningConfig,
)


def _set_patch(module, module_name: str) -> None:
    """Replace PEFT's module-level symbols with the MetaPrefix implementations."""
    setattr(module, "PrefixEncoder", MetaPrefixEncoder)
    setattr(module, "PrefixTuningConfig", MetaPrefixTuningConfig)
    print(f"[MetaPrefix patch] patched {module_name}")


# The implementation module exports these names.
prefix_tuning_module = importlib.import_module("peft.tuners.prefix_tuning")
_set_patch(prefix_tuning_module, "peft.tuners.prefix_tuning")

# PeftModel commonly imports PrefixEncoder into its own module namespace at
# import time. Patching only peft.tuners.prefix_tuning may therefore leave the
# already-imported alias unchanged.
peft_model_module = importlib.import_module("peft.peft_model")
_set_patch(peft_model_module, "peft.peft_model")

# Also expose the custom classes from the top-level peft namespace, matching
# the behavior of the original project patch.
_set_patch(peft, "peft")

# Patch any other already-imported PEFT modules that copied these symbols.
for _module_name, _module in tuple(sys.modules.items()):
    if not _module_name.startswith("peft.") or _module is None:
        continue
    if hasattr(_module, "PrefixEncoder"):
        setattr(_module, "PrefixEncoder", MetaPrefixEncoder)
    if hasattr(_module, "PrefixTuningConfig"):
        setattr(_module, "PrefixTuningConfig", MetaPrefixTuningConfig)

# Fail early if the main PEFT construction path still points elsewhere.
assert prefix_tuning_module.PrefixEncoder is MetaPrefixEncoder
assert peft_model_module.PrefixEncoder is MetaPrefixEncoder
assert peft.PrefixEncoder is MetaPrefixEncoder
assert prefix_tuning_module.PrefixTuningConfig is MetaPrefixTuningConfig
assert peft_model_module.PrefixTuningConfig is MetaPrefixTuningConfig
assert peft.PrefixTuningConfig is MetaPrefixTuningConfig

print("[MetaPrefix patch] SUCCESS: custom PrefixEncoder and PrefixTuningConfig are active.")
print(f"[MetaPrefix patch] PEFT version: {getattr(peft, '__version__', 'unknown')}")
