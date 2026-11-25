# -*- coding: utf-8 -*-
"""Stage 2: target-cancer FOMAML adaptation from the Stage 1 initialization."""
from __future__ import annotations
import argparse, json, re, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, get_polynomial_decay_schedule_with_warmup
from common import (ROOT, DEFAULT_OUT, DEFAULT_MODEL, ZERO_SHOT_TASKS,
                    read_csv, get_meta_tasks, make_loader, next_batch,
                    move_batch, trainable_param_dict, loss_with_params,
                    build_adapted_model,
                    save_adapter, full_support_fast_params, evaluate_fast_params,
                    apply_fast_params, check_partition_integrity,
                    set_seed, device_from_arg)


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True, help="Exact cancer column name, e.g. Breast cancer")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--stage1", type=Path, default=ROOT / "models" / "stage1" / "best")
    p.add_argument("--out", type=Path, default=None,
                   help="Directory for the Validation-selected FOMAML initialization (meta_init).")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--steps", type=int, default=20,
                   help="FOMAML outer updates; manuscript method specifies 20 adaptation steps.")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-length", type=int, default=50)
    p.add_argument("--inner-lr", type=float, default=1e-4)
    p.add_argument("--outer-lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    if args.target in ZERO_SHOT_TASKS:
        raise ValueError(
            f"{args.target} is a held-out Zero-shot task. Do not adapt it. "
            "Use the Stage 1 adapter directly for zero-shot transfer."
        )
    set_seed(args.seed)
    device = device_from_arg(args.device)

    train = read_csv(args.data_dir / "train.csv")
    valid = read_csv(args.data_dir / "valid.csv")
    # Keep the final Test set unopened until the target adapter is finalized.
    check_partition_integrity(train, valid)
    meta_support = read_csv(args.data_dir / "meta_support.csv")
    meta_query = read_csv(args.data_dir / "meta_query.csv")
    tasks = get_meta_tasks(args.data_dir / "meta_task_manifest.csv")
    if args.target not in tasks:
        raise ValueError(
            f"Target task {args.target!r} is not included in Stage 1 task manifest. "
            "Check meta_task_manifest.csv and the independent group counts."
        )

    support_df = meta_support[meta_support[args.target] == 1]
    query_df = meta_query[meta_query[args.target] == 1]
    valid_df = valid[valid[args.target] == 1]
    if support_df.empty or query_df.empty:
        raise RuntimeError("Target lacks group-disjoint Stage 2 meta-support/meta-query data.")
    if valid_df.empty:
        raise RuntimeError("Target has no positive sequences in Validation; cannot select Stage 2 checkpoint.")
    if set(support_df.split_group_id.astype(str)) & set(query_df.split_group_id.astype(str)):
        raise RuntimeError("Target Stage 2 support/query group leakage detected.")
    train_ids = set(train.peptide_id.astype(str))
    target_role_ids = set(support_df.peptide_id.astype(str)) | set(query_df.peptide_id.astype(str))
    if not target_role_ids.issubset(train_ids):
        raise RuntimeError("Stage 2 support/query contain sequences outside global Train.")

    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    support_loader = make_loader(
        support_df, tokenizer, args.batch_size, args.max_length, True, args.seed,
        context_mode="random",
    )
    support_eval_loader = make_loader(
        support_df, tokenizer, args.batch_size, args.max_length, False, args.seed,
        context_mode="random", context_seed=args.seed,
    )
    query_loader = make_loader(
        query_df, tokenizer, args.batch_size, args.max_length, True, args.seed + 1,
        context_mode="random",
    )
    # Validation and final Test measure de-novo generation from the EOS sentinel.
    valid_loader = make_loader(
        valid_df, tokenizer, args.batch_size, args.max_length, False, args.seed + 2,
        context_mode="zero",
    )

    model = build_adapted_model(args.model, args.stage1, device, trainable=True)
    model.config.use_cache = False
    model.train()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.outer_lr, weight_decay=args.weight_decay
    )
    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=args.steps,
        lr_end=1e-10, power=3
    )
    iterator_state = {}
    output_dir = args.out or (ROOT / "models" / "stage2" / slug(args.target) / "meta_init")
    history, best_val, best_step, bad_steps = [], float("inf"), 0, 0
    start_time = time.time()

    for step in range(1, args.steps + 1):
        model.train()
        step_start = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        support_batch = move_batch(next_batch(support_loader, iterator_state, "support"), device)
        query_batch = move_batch(next_batch(query_loader, iterator_state, "query"), device)

        # FOMAML: detach support gradients so d(theta')/d(theta) is approximated
        # by identity. The query gradient is then applied to the Stage 1 init.
        optimizer.zero_grad(set_to_none=True)
        params, trainable = trainable_param_dict(model)
        buffers = dict(model.named_buffers())
        support_loss = loss_with_params(model, params, buffers, support_batch)
        names = list(trainable)
        grads = torch.autograd.grad(
            support_loss,
            [trainable[n] for n in names],
            create_graph=False,
            allow_unused=True,
        )
        fast_params = dict(params)
        for name, grad in zip(names, grads):
            if grad is not None:
                fast_params[name] = params[name] - args.inner_lr * grad.detach()
        query_loss = loss_with_params(model, fast_params, buffers, query_batch)
        query_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
        optimizer.step()
        scheduler.step()
        train_step_seconds = time.time() - step_start
        if device.type == "cuda":
            peak_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        else:
            peak_allocated_mb = float("nan")
            peak_reserved_mb = float("nan")

        # Validation: adapt once using training support, then evaluate on
        # independent Validation sequences; test is not touched here.
        # Validation evaluates the target-adapted fast weights derived from
        # the complete Train Support partition (not from Test).
        val_fast, val_buffers = full_support_fast_params(
            model, support_eval_loader, args.inner_lr, device
        )
        val_loss = evaluate_fast_params(
            model, val_fast, val_buffers, valid_loader, device
        )
        history.append({
            "step": step,
            "support_loss": float(support_loss.detach().item()),
            "meta_query_loss": float(query_loss.detach().item()),
            "validation_adapted_loss": val_loss,
            "train_step_seconds": train_step_seconds,
            "peak_cuda_allocated_mb": peak_allocated_mb,
            "peak_cuda_reserved_mb": peak_reserved_mb,
            "lr": scheduler.get_last_lr()[0],
        })
        print(
            f"{args.target} | Step {step:02d}/{args.steps} | "
            f"support={support_loss.item():.5f} | query={query_loss.item():.5f} | "
            f"val_adapted={val_loss:.5f} | step={train_step_seconds:.2f}s | "
            f"peak_alloc={peak_allocated_mb:.1f}MB"
        )

        if val_loss < best_val:
            best_val, best_step, bad_steps = val_loss, step, 0
            save_adapter(model, tokenizer, output_dir, vars_to_json(args))
        else:
            bad_steps += 1
        if bad_steps >= args.patience:
            print(f"Early stopping at step {step}; best step={best_step}.")
            break

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(output_dir.parent / "stage2_history.csv", index=False)

    # Reload the best Validation-selected FOMAML initialization.
    # Apply one final inner update on the full target Support set and save
    # target_adapter. This is the adapter used for target-specific generation.
    model = build_adapted_model(args.model, output_dir, device, trainable=True)
    target_fast, target_buffers = full_support_fast_params(
        model, support_eval_loader, args.inner_lr, device
    )
    apply_fast_params(model, target_fast)
    target_adapter_dir = output_dir.parent / "target_adapter"
    save_adapter(model, tokenizer, target_adapter_dir,
                 {**vars_to_json(args), "final_support_adaptation": "one token-weighted first-order update over full meta_support"})

    # Open the final Test file only after checkpoint selection and final Support adaptation.
    test = read_csv(args.data_dir / "test.csv")
    check_partition_integrity(train, valid, test)
    test_df = test[test[args.target] == 1]
    if not test_df.empty:
        test_loader = make_loader(
            test_df, tokenizer, args.batch_size, args.max_length, False, args.seed + 3,
            context_mode="zero",
        )
        test_loss = evaluate_fast_params(
            model, dict(model.named_parameters()), dict(model.named_buffers()),
            test_loader, device
        )
        test_n = len(test_df)
    else:
        test_loss, test_n = None, 0
        print("WARNING: target has no positive sequences in Test; no held-out test loss can be reported.")
    summary = {
        "target_cancer": args.target,
        "meta_init_dir": str(output_dir),
        "target_adapter_dir": str(target_adapter_dir),
        "best_step": best_step,
        "best_validation_adapted_loss": best_val,
        "test_adapted_loss": test_loss,
        "test_positive_sequences": test_n,
        "train_support_sequences": len(support_df),
        "train_meta_query_sequences": len(query_df),
        "validation_positive_sequences": len(valid_df),
        "seed": args.seed,
        "elapsed_seconds": time.time() - start_time,
        "test_evaluated_once_after_checkpoint_selection": True,
    }
    (output_dir.parent / "stage2_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def vars_to_json(args):
    return {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}

if __name__ == "__main__":
    main()
