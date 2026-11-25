# -*- coding: utf-8 -*-
"""Generate ACP candidates with a Stage 1 or cancer-adapted Stage 2 adapter."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer

from common import ROOT, DEFAULT_MODEL, build_adapted_model, device_from_arg, set_seed

AA = set("ACDEFGHIKLMNPQRSTVWY")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=Path, required=True,
                        help="Stage 1 best adapter or Stage 2 target_adapter directory.")
    parser.add_argument("--target", default="general",
                        help="Target label saved in the output table; Stage 2 adapters are target-specific.")
    parser.add_argument("--prompt", default="",
                        help="Optional observed N-terminal amino-acid context; leave empty for de-novo generation.")
    parser.add_argument("--out", type=Path, default=ROOT / "output" / "generated.csv")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--min-new-tokens", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.n < 1:
        raise ValueError("--n must be >= 1.")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be >= 1.")
    prompt = re.sub(r"\s+", "", args.prompt).upper()
    if any(char not in AA for char in prompt):
        raise ValueError("--prompt must contain only the 20 standard amino-acid letters.")

    set_seed(args.seed)
    device = device_from_arg(args.device)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = build_adapted_model(args.model, args.adapter, device, trainable=False)
    model.eval()

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"] if prompt else []
    condition_ids = [int(tokenizer.eos_token_id)] + [int(x) for x in prompt_ids]
    decoder_start_ids = condition_ids
    decoder_start_length = len(decoder_start_ids)

    batch_size = min(32, max(1, args.n))
    max_attempts = max(args.n * 8, args.n + 100)
    collected: list[str] = []
    seen: set[str] = set()
    attempts = 0

    while len(collected) < args.n and attempts < max_attempts:
        current_batch = min(batch_size, args.n - len(collected))
        cond = torch.tensor([condition_ids] * current_batch, dtype=torch.long, device=device)
        decoder = torch.tensor(
            [decoder_start_ids] * current_batch, dtype=torch.long, device=device
        )
        with torch.inference_mode():
            generated = model.generate(
                condition_input_ids=cond,
                decoder_input_ids=decoder,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.min_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True,
            )
        for token_row in generated:
            new_ids = token_row[decoder_start_length:].tolist()
            generated_suffix = tokenizer.decode(new_ids, skip_special_tokens=True)
            generated_suffix = re.sub(r"\s+", "", generated_suffix).upper()
            sequence = prompt + generated_suffix
            if (
                sequence
                and 5 <= len(sequence) <= 50
                and all(char in AA for char in sequence)
                and sequence not in seen
            ):
                seen.add(sequence)
                collected.append(sequence)
                if len(collected) >= args.n:
                    break
        attempts += current_batch

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "sequence_id": [f"GEN{i + 1:06d}" for i in range(len(collected))],
        "Sequence": collected,
        "target": args.target,
        "adapter": str(args.adapter),
        "prompt_context": prompt,
        "seed": args.seed,
    }).to_csv(args.out, index=False)
    print(f"Generated {len(collected)}/{args.n} unique valid peptides -> {args.out}")
    if len(collected) < args.n:
        print("WARNING: maximum attempts reached before reaching the requested count.")


if __name__ == "__main__":
    main()
