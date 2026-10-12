# generate.py
import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import tools.patch_peft  # noqa: F401

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

from utils.set_seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--peft_model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="generated.npy")
    parser.add_argument("--num_sequences", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=50)
    parser.add_argument("--top_k", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    config = PeftConfig.from_pretrained(args.peft_model_path)
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(model, args.peft_model_path)

    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name_or_path,
        padding_side="left",
    )

    model.to(args.device)
    model.eval()

    input_ids = tokenizer(
        ["<|endoftext|>"],
        return_tensors="pt",
    )["input_ids"].to(args.device)

    seq_list = []

    with torch.no_grad():
        pbar = tqdm(total=args.num_sequences)
        while len(seq_list) < args.num_sequences:
            cur_bs = min(args.batch_size, args.num_sequences - len(seq_list))

            outputs = model.generate(
                input_ids=input_ids,
                max_length=args.max_length,
                do_sample=True,
                top_k=args.top_k,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                num_return_sequences=cur_bs,
                eos_token_id=tokenizer.eos_token_id,
            )

            seqs = tokenizer.batch_decode(
                outputs.detach().cpu().numpy(),
                skip_special_tokens=True,
            )
            seq_list.extend(seqs)
            pbar.update(cur_bs)

        pbar.close()

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    np.save(args.output_path, np.array(seq_list, dtype=object))
    print(f"Saved {len(seq_list)} sequences to {args.output_path}")


if __name__ == "__main__":
    main()