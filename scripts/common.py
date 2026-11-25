# -*- coding: utf-8 -*-
"""Shared utilities for reproducible MetaPrefix training."""
from __future__ import annotations

import json
import random
import hashlib
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.func import functional_call

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_MODEL = ROOT / "ProtGPT2"
DEFAULT_OUT = ROOT / "output"

CANCER_COLS = [
    "Breast cancer",
    "Lung cancer",
    "Hepatocellular carcinoma (HCC)",
    "Liver cancer",
    "Melanoma",
    "Colon adenocarcinoma",
    "Ovarian cancer",
    "Cervical cancer",
    "Kidney cancer",
    "Pancreatic cancer",
    "Prostate cancer",
    "Central nervous system glioma",
    "Cutaneous squamous cell carcinoma (cSCC)",
    "Bladder cancer",
    "Lymphoma",
    "Neuroblastoma",
    "Rhabdomyosarcoma",
    "Multiple myeloma",
    "Fibrosarcoma",
    "Head and neck squamous cell carcinoma (HNSCC)",
    "Gastric cancer",
    "Leukemia",
    "Osteosarcoma",
    "Esophageal cancer",
    "Endometrial cancer",
    "Nasopharyngeal carcinoma (NPC)",
    "Pleural mesothelioma",
    "Retinoblastoma",
]

# Explicitly held-out tasks. These are not used for Stage 1 or Stage 2 adaptation.
ZERO_SHOT_TASKS = [
    "Endometrial cancer",
    "Esophageal cancer",
    "Nasopharyngeal carcinoma (NPC)",
    "Osteosarcoma",
    "Pleural mesothelioma",
    "Retinoblastoma",
]

META_COLUMNS = {
    "peptide_id", "Sequence", "cluster_id", "split_group_id",
    "group_split", "dataset", "Dataset", "_meta_role",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Determinism is preferred for reproducibility; some CUDA kernels may still
    # vary slightly across hardware / library versions.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def device_from_arg(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    df = pd.read_csv(path, dtype={"peptide_id": str, "Sequence": str,
                                  "cluster_id": str, "split_group_id": str})
    required = {"Sequence", "peptide_id", "cluster_id", "split_group_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    df["Sequence"] = df["Sequence"].astype(str).str.upper().str.strip()
    bad = ~df["Sequence"].str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+", na=False)
    if bad.any():
        raise ValueError(f"{path.name}: {int(bad.sum())} sequences contain non-standard amino acids.")
    if df["Sequence"].duplicated().any():
        raise ValueError(f"{path.name}: duplicate exact sequences remain; deduplicate before training.")
    if df["split_group_id"].isna().any():
        raise ValueError(f"{path.name}: missing split_group_id values.")
    df["split_group_id"] = df["split_group_id"].astype(str)
    for col in CANCER_COLS:
        if col not in df.columns:
            raise ValueError(f"{path.name}: missing cancer-label column {col!r}.")
        df[col] = pd.to_numeric(df[col], errors="raise").astype(int)
        if not set(df[col].unique()).issubset({0, 1}):
            raise ValueError(f"{path.name}: {col!r} must be binary 0/1.")
    return df


def check_partition_integrity(train: pd.DataFrame, valid: pd.DataFrame,
                              test: pd.DataFrame | None = None,
                              zero: pd.DataFrame | None = None,
                              buffer: pd.DataFrame | None = None) -> None:
    parts = {"Train": train, "Validation": valid}
    if test is not None:
        parts["Test"] = test
    if zero is not None:
        parts["Zero-shot"] = zero
    if buffer is not None:
        parts["ZS-buffer"] = buffer
    names = list(parts)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            # Zero-shot core and ZS-buffer intentionally may share similarity groups.
            if {a, b} == {"Zero-shot", "ZS-buffer"}:
                continue
            ga = set(parts[a]["split_group_id"].astype(str))
            gb = set(parts[b]["split_group_id"].astype(str))
            overlap = ga & gb
            if overlap:
                raise RuntimeError(f"split_group_id leakage: {a} vs {b}: {len(overlap)} shared groups.")
    for part_name in ("Train", "Validation", "Test"):
        if part_name not in parts:
            continue
        part = parts[part_name]
        for col in ZERO_SHOT_TASKS:
            if int(part[col].sum()):
                raise RuntimeError(
                    f"Held-out Zero-shot label {col!r} appears in {part_name}."
                )


def get_meta_tasks(manifest_path: Path) -> List[str]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing task manifest {manifest_path}; run 04_prepare_meta.py first.")
    mf = pd.read_csv(manifest_path)
    tasks = mf.loc[mf["Stage1_status"] == "include", "Cancer"].astype(str).tolist()
    tasks = [t for t in tasks if t not in ZERO_SHOT_TASKS]
    if not tasks:
        raise RuntimeError("No cancer tasks passed Stage 1 group-count requirements.")
    return tasks


class PeptideDataset(Dataset):
    def __init__(self, sequences: Iterable[str]):
        self.sequences = list(sequences)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index]


def make_collate(
    tokenizer,
    max_length: int,
    *,
    context_mode: str = "random",
    context_seed: int | None = None,
    context_max_tokens: int = 8,
    zero_context_probability: float = 0.5,
):
    """Build causal decoder inputs plus a separate observed N-terminal context.

    A context contains EOS and zero or more *known preceding* peptide tokens.
    The decoder receives the complete observed prefix followed by the target suffix;
    context positions are masked from the loss. This preserves residue order in the
    causal decoder while the Cross-Attention prefix sees no future target residues.
    """
    if tokenizer.eos_token_id is None:
        raise ValueError("ProtGPT2 tokenizer must define eos_token_id.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if max_length < 4:
        raise ValueError("max_length must be >= 4 to retain a start token and peptide target.")
    if context_mode not in {"random", "zero"}:
        raise ValueError("context_mode must be 'random' or 'zero'.")
    if context_max_tokens < 0:
        raise ValueError("context_max_tokens must be >= 0.")
    if not 0.0 <= zero_context_probability <= 1.0:
        raise ValueError("zero_context_probability must be in [0, 1].")

    eos_id = int(tokenizer.eos_token_id)
    pad_id = int(tokenizer.pad_token_id)

    def choose_context_length(sequence: str, n_tokens: int) -> int:
        max_k = min(int(context_max_tokens), max(0, n_tokens - 1))
        if context_mode == "zero" or max_k == 0:
            return 0
        if context_seed is None:
            rng = random
        else:
            digest = hashlib.sha256(f"{context_seed}::{sequence}".encode("utf-8")).digest()
            rng = random.Random(int.from_bytes(digest[:8], "big", signed=False))
        if rng.random() < zero_context_probability:
            return 0
        return rng.randint(1, max_k)

    def collate(sequences: List[str]) -> Dict[str, torch.Tensor]:
        encoded = tokenizer(
            sequences,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length - 2,
            padding=False,
        )
        decoder_ids_batch, decoder_mask_batch, labels_batch = [], [], []
        context_ids_batch, context_mask_batch = [], []

        for sequence, raw_ids in zip(sequences, encoded["input_ids"]):
            aa_ids = [int(x) for x in raw_ids[: max_length - 2]]
            if not aa_ids:
                raise ValueError(f"Tokenizer returned no amino-acid tokens for sequence {sequence!r}.")

            context_len = choose_context_length(sequence, len(aa_ids))
            context_ids = [eos_id] + aa_ids[:context_len]
            suffix_ids = aa_ids[context_len:] + [eos_id]
            decoder_ids = context_ids + suffix_ids
            labels = [-100] * len(context_ids) + suffix_ids

            if len(decoder_ids) > max_length:
                raise RuntimeError("Internal sequence construction exceeded max_length.")
            decoder_mask = [1] * len(decoder_ids)
            context_mask = [1] * len(context_ids)

            decoder_ids_batch.append(decoder_ids)
            decoder_mask_batch.append(decoder_mask)
            labels_batch.append(labels)
            context_ids_batch.append(context_ids)
            context_mask_batch.append(context_mask)

        max_decoder = max(map(len, decoder_ids_batch))
        max_context = max(map(len, context_ids_batch))
        for i in range(len(sequences)):
            pad_decoder = max_decoder - len(decoder_ids_batch[i])
            pad_context = max_context - len(context_ids_batch[i])
            decoder_ids_batch[i] += [pad_id] * pad_decoder
            decoder_mask_batch[i] += [0] * pad_decoder
            labels_batch[i] += [-100] * pad_decoder
            context_ids_batch[i] += [pad_id] * pad_context
            context_mask_batch[i] += [0] * pad_context

        return {
            "input_ids": torch.tensor(decoder_ids_batch, dtype=torch.long),
            "attention_mask": torch.tensor(decoder_mask_batch, dtype=torch.long),
            "labels": torch.tensor(labels_batch, dtype=torch.long),
            "condition_input_ids": torch.tensor(context_ids_batch, dtype=torch.long),
            "condition_attention_mask": torch.tensor(context_mask_batch, dtype=torch.long),
        }

    return collate


def make_loader(
    df: pd.DataFrame,
    tokenizer,
    batch_size: int,
    max_length: int,
    shuffle: bool,
    seed: int = 42,
    *,
    context_mode: str = "random",
    context_seed: int | None = None,
) -> DataLoader:
    if df.empty:
        raise ValueError("Cannot make a DataLoader from an empty peptide set.")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        PeptideDataset(df["Sequence"].astype(str).tolist()),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=make_collate(
            tokenizer,
            max_length,
            context_mode=context_mode,
            context_seed=context_seed,
        ),
        generator=generator,
        drop_last=False,
    )

def next_batch(loader: DataLoader, state: dict, key: str):
    iterator = state.get(key)
    if iterator is None:
        iterator = iter(loader)
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    state[key] = iterator
    return batch


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def loss_with_params(model, params, buffers, batch):
    out = functional_call(model, (params, buffers), args=(), kwargs=batch)
    return out.loss


def trainable_param_dict(model):
    params = dict(model.named_parameters())
    trainable = {name: p for name, p in params.items() if p.requires_grad}
    if not trainable:
        raise RuntimeError(
            "No trainable MetaPrefix parameters found. Check adapter construction and loading."
        )
    return params, trainable


def inner_adapt(model, batch, inner_lr: float, second_order: bool):
    """One differentiable SGD step. FOMAML detaches support gradients only."""
    params, trainable = trainable_param_dict(model)
    buffers = dict(model.named_buffers())
    support_loss = loss_with_params(model, params, buffers, batch)
    names = list(trainable)
    grads = torch.autograd.grad(
        support_loss,
        [trainable[n] for n in names],
        create_graph=second_order,
        allow_unused=True,
    )
    fast = dict(params)
    for name, grad in zip(names, grads):
        if grad is not None:
            update = grad if second_order else grad.detach()
            fast[name] = params[name] - inner_lr * update
    return fast, buffers, support_loss.detach()


def adapted_eval_loss(model, support_batch, query_loader, inner_lr: float,
                      device: torch.device, max_query_batches: int | None = None) -> float:
    """Evaluate a one-step adapted model without updating the meta-initialization."""
    model.eval()
    support_batch = move_batch(support_batch, device)
    with torch.enable_grad():
        fast, buffers, _ = inner_adapt(model, support_batch, inner_lr, second_order=False)
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for i, batch in enumerate(query_loader):
            if max_query_batches is not None and i >= max_query_batches:
                break
            batch = move_batch(batch, device)
            out = functional_call(model, (fast, buffers), args=(), kwargs=batch)
            n_tokens = int((batch["labels"] != -100).sum().item())
            total_loss += float(out.loss.item()) * max(n_tokens, 1)
            total_tokens += n_tokens
    return total_loss / max(total_tokens, 1)


def save_adapter(model, tokenizer, out_dir: Path, config_dict: dict):
    """Save the trainable MetaPrefix encoder and tokenizer (not the frozen LM)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, run_config=config_dict)
    tokenizer.save_pretrained(out_dir)


def build_stage1_model(
    base_model_path: Path,
    num_virtual_tokens: int,
    encoder_hidden_size: int,
    device: torch.device,
):
    """Create ProtGPT2 + the trainable MetaPrefix encoder."""
    from transformers import AutoModelForCausalLM
    from peft_overlay import MetaPrefixConfig, MetaPrefixForCausalLM

    # Eager attention is intentionally selected for higher-order autograd.
    # Fused SDPA/Flash backends may not implement the double backward needed
    # by Stage 1's second-order MAML query gradient.
    base = AutoModelForCausalLM.from_pretrained(
        str(base_model_path)
    )
    base.config.use_cache = False
    prefix_config = MetaPrefixConfig.from_transformer_config(
        base.config,
        num_virtual_tokens=num_virtual_tokens,
        encoder_hidden_size=encoder_hidden_size,
    )
    model = MetaPrefixForCausalLM(base, prefix_config)
    model.to(device)
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable parameters: {trainable_count:,} / {total_count:,} "
        f"({100 * trainable_count / max(total_count, 1):.4f}%)"
    )
    return model


def build_adapted_model(
    base_model_path: Path,
    adapter_path: Path,
    device: torch.device,
    *,
    trainable: bool = True,
):
    """Load a saved MetaPrefix adapter onto a fresh frozen ProtGPT2 backbone."""
    from transformers import AutoModelForCausalLM
    from peft_overlay import MetaPrefixForCausalLM

    base = AutoModelForCausalLM.from_pretrained(
        str(base_model_path)
    )
    base.config.use_cache = False
    model = MetaPrefixForCausalLM.from_pretrained(
        base,
        adapter_path,
        trainable=trainable,
    )
    model.to(device)
    return model

def full_support_fast_params(model, support_loader, inner_lr: float,
                             device: torch.device):
    """One first-order inner update using the token-weighted full Support set."""
    model.eval()
    params, trainable = trainable_param_dict(model)
    buffers = dict(model.named_buffers())
    names = list(trainable)
    grad_sums = {n: torch.zeros_like(trainable[n]) for n in names}
    total_tokens = 0
    for batch in support_loader:
        batch = move_batch(batch, device)
        loss = loss_with_params(model, params, buffers, batch)
        grads = torch.autograd.grad(
            loss, [trainable[n] for n in names],
            create_graph=False, allow_unused=True
        )
        n_tokens = int((batch["labels"] != -100).sum().item())
        total_tokens += n_tokens
        for name, grad in zip(names, grads):
            if grad is not None:
                grad_sums[name].add_(grad.detach(), alpha=max(n_tokens, 1))
        del loss, grads
    if total_tokens == 0:
        raise RuntimeError("No valid amino-acid tokens in full Support set.")
    fast = dict(params)
    for name in names:
        fast[name] = params[name] - inner_lr * (grad_sums[name] / total_tokens)
    return fast, buffers


def evaluate_fast_params(model, fast_params, buffers, query_loader,
                         device: torch.device) -> float:
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for batch in query_loader:
            batch = move_batch(batch, device)
            out = functional_call(model, (fast_params, buffers), args=(), kwargs=batch)
            n_tokens = int((batch["labels"] != -100).sum().item())
            total_loss += float(out.loss.item()) * max(n_tokens, 1)
            total_tokens += n_tokens
    return total_loss / max(total_tokens, 1)


def apply_fast_params(model, fast_params):
    """Materialize fast weights as the model's adapter parameters for saving/inference."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad and name in fast_params:
                param.copy_(fast_params[name].detach())
