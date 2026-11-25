# -*- coding: utf-8 -*-
"""Preflight checks for the self-contained MetaPrefix project."""
from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

print("Project:", ROOT)
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
_torch_version = torch.__version__.split("+")[0].split(".")
if tuple(int(x) for x in _torch_version[:2]) < (2, 1):
    raise SystemExit("PyTorch >=2.1 is required (torch.func.functional_call is used).")
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

required = ["transformers", "numpy", "pandas"]
missing = []
for package in required:
    installed = importlib.util.find_spec(package) is not None
    print(f"{package}:", "installed" if installed else "MISSING")
    if not installed:
        missing.append(package)
print("MMseqs2:", shutil.which("mmseqs") or "MISSING from PATH (required only for Step A)")
print("Input CSV:", (ROOT / "input" / "ACP_final_merged_with_cluster.csv").exists())
print("ProtGPT2 folder:", (ROOT / "ProtGPT2").exists())
print("Legacy PEFT monkey patch:", "not required (self-contained MetaPrefix wrapper)")

if (ROOT / "input" / "ACP_final_merged_with_cluster.csv").exists():
    df = pd.read_csv(ROOT / "input" / "ACP_final_merged_with_cluster.csv")
    print("Input shape:", df.shape)
    print("Columns:", len(df.columns))

try:
    from peft_overlay import MetaPrefixConfig, MetaPrefixForCausalLM, PrefixEncoder
    print("MetaPrefix package: import OK")
except Exception as exc:
    print("MetaPrefix package: IMPORT FAILED:", repr(exc))
    raise

if missing:
    raise SystemExit("Missing required packages: " + ", ".join(missing) + ". Install requirements.txt first.")
