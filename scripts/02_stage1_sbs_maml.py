# -*- coding: utf-8 -*-
"""Stage 1: sequential second-order MAML over cancer-specific ACP tasks."""
from __future__ import annotations
import argparse, json, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, get_polynomial_decay_schedule_with_warmup
from common import (ROOT, DEFAULT_OUT, DEFAULT_MODEL, ZERO_SHOT_TASKS,
                    read_csv, get_meta_tasks, make_loader, next_batch,
                    move_batch, trainable_param_dict, loss_with_params,
                    full_support_fast_params, evaluate_fast_params,
                    check_partition_integrity, build_stage1_model, save_adapter,
                    set_seed, device_from_arg)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--out", type=Path, default=ROOT / "models" / "stage1" / "best")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--steps", type=int, default=100,
                   help="Outer meta-updates; manuscript method specifies 100 iterations.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=50)
    p.add_argument("--num-virtual-tokens", type=int, default=10)
    p.add_argument("--encoder-hidden-size", type=int, default=1280)
    p.add_argument("--inner-lr", type=float, default=1e-4)
    p.add_argument("--outer-lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--max-tasks", type=int, default=0,
                   help="Optional task cap for K-scaling profiling; 0 means all eligible tasks.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def eval_meta_validation(model, support_df, valid_df, tasks, tokenizer,
                          batch_size, max_length, inner_lr, device, seed):
    losses = []
    for task in tasks:
        sup = support_df[support_df[task] == 1]
        qry = valid_df[valid_df[task] == 1]
        if sup.empty or qry.empty:
            continue
        # Fixed support ordering makes validation checkpoint selection stable.
        sup_loader = make_loader(
            sup, tokenizer, batch_size, max_length, False, seed,
            context_mode="random", context_seed=seed,
        )
        # Validation NLL is evaluated in the de-novo setting (EOS-only context).
        qry_loader = make_loader(
            qry, tokenizer, batch_size, max_length, False, seed,
            context_mode="zero",
        )
        # Use every positive Train meta-support peptide for the validation-time
        # first-order adaptation; Validation remains query-only and group-disjoint.
        fast_params, buffers = full_support_fast_params(
            model, sup_loader, inner_lr, device
        )
        loss = evaluate_fast_params(
            model, fast_params, buffers, qry_loader, device
        )
        if np.isfinite(loss):
            losses.append(loss)
    if not losses:
        raise RuntimeError(
            "No Stage 1 task has both Train meta-support and Validation positives. "
            "Review meta_task_manifest.csv and the validation split."
        )
    return float(np.mean(losses)), len(losses)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = device_from_arg(args.device)

    train = read_csv(args.data_dir / "train.csv")
    valid = read_csv(args.data_dir / "valid.csv")
    meta_support = read_csv(args.data_dir / "meta_support.csv")
    meta_query = read_csv(args.data_dir / "meta_query.csv")
    # The final Test file is deliberately not opened by Stage 1. Its split
    # integrity must already have passed in scripts/03_audit_data.py.
    check_partition_integrity(train, valid)
    tasks = get_meta_tasks(args.data_dir / "meta_task_manifest.csv")

    train_ids = set(train.peptide_id.astype(str))
    role_ids = set(meta_support.peptide_id.astype(str)) | set(meta_query.peptide_id.astype(str))
    if train_ids != role_ids:
        raise RuntimeError("meta_support/meta_query do not form an exact partition of global Train.")
    if set(meta_support.split_group_id.astype(str)) & set(meta_query.split_group_id.astype(str)):
        raise RuntimeError("Meta-support and Meta-query share split_group_id values.")

    # Hard checks: six held-out tasks cannot enter Stage 1.
    for task in ZERO_SHOT_TASKS:
        if train[task].sum() or valid[task].sum() or meta_support[task].sum() or meta_query[task].sum():
            raise RuntimeError(f"Held-out Zero-shot task leaked into Stage 1: {task}")
    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    task_loaders = {}
    for task in tasks:
        sup = meta_support[meta_support[task] == 1]
        qry = meta_query[meta_query[task] == 1]
        if sup.empty or qry.empty:
            continue
        task_loaders[task] = (
            make_loader(
                sup, tokenizer, args.batch_size, args.max_length, True, args.seed,
                context_mode="random",
            ),
            make_loader(
                qry, tokenizer, args.batch_size, args.max_length, True, args.seed + 1,
                context_mode="random",
            ),
        )
    if len(task_loaders) < 2:
        raise RuntimeError("SBS-MAML requires at least two valid meta-training cancer tasks.")
    tasks = list(task_loaders)
    if args.max_tasks > 0:
        if args.max_tasks < 2:
            raise ValueError("--max-tasks must be 0 (all tasks) or >=2 for meta-learning.")
        tasks = tasks[:min(args.max_tasks, len(tasks))]
        print(f"Task-cap profiling mode: using K={len(tasks)} tasks")

    model = build_stage1_model(
        args.model, args.num_virtual_tokens, args.encoder_hidden_size, device
    )
    model.config.use_cache = False
    model.train()

    trainable_names = [n for n,p in model.named_parameters() if p.requires_grad]
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.outer_lr, weight_decay=args.weight_decay
    )
    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=args.steps,
        lr_end=1e-10, power=3
    )

    # Each task has an independent iterator. One Support and one Meta-query batch
    # are processed per cancer task at each outer update, sequentially.
    iterator_state = {}
    history = []
    best_val, best_step, bad_steps = float("inf"), 0, 0
    args.out.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        meta_step_start = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        order = tasks.copy()
        random.shuffle(order)
        meta_losses, support_losses = [], []

        # SBS: do not retain K task graphs simultaneously. Backpropagate and
        # release each task graph before moving to the next task.
        for task in order:
            sup_loader, qry_loader = task_loaders[task]
            sup_batch = move_batch(next_batch(sup_loader, iterator_state, task + "_s"), device)
            qry_batch = move_batch(next_batch(qry_loader, iterator_state, task + "_q"), device)

            params, trainable = trainable_param_dict(model)
            buffers = dict(model.named_buffers())
            support_loss = loss_with_params(model, params, buffers, sup_batch)
            names = list(trainable)
            grads = torch.autograd.grad(
                support_loss,
                [trainable[n] for n in names],
                create_graph=True,      # true second-order MAML
                allow_unused=True,
            )
            fast_params = dict(params)
            for name, grad in zip(names, grads):
                if grad is not None:
                    fast_params[name] = params[name] - args.inner_lr * grad

            query_loss = loss_with_params(model, fast_params, buffers, qry_batch)
            (query_loss / len(order)).backward()
            meta_losses.append(float(query_loss.detach().item()))
            support_losses.append(float(support_loss.detach().item()))

            del params, trainable, buffers, support_loss, grads, fast_params, query_loss

        torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
        optimizer.step()
        scheduler.step()
        meta_step_seconds = time.time() - meta_step_start
        if device.type == "cuda":
            peak_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        else:
            peak_allocated_mb = float("nan")
            peak_reserved_mb = float("nan")

        train_meta = float(np.mean(meta_losses))
        train_support = float(np.mean(support_losses))
        val_meta, val_task_n = eval_meta_validation(
            model, meta_support, valid, tasks, tokenizer,
            args.batch_size, args.max_length, args.inner_lr, device, args.seed
        )
        history.append({
            "step": step, "train_meta_query_loss": train_meta,
            "train_support_loss": train_support,
            "validation_adapted_query_loss": val_meta,
            "validation_tasks": val_task_n,
            "num_meta_tasks_K": len(order),
            "meta_step_seconds": meta_step_seconds,
            "peak_cuda_allocated_mb": peak_allocated_mb,
            "peak_cuda_reserved_mb": peak_reserved_mb,
            "lr": scheduler.get_last_lr()[0],
        })
        print(
            f"Step {step:03d}/{args.steps} | train_meta={train_meta:.5f} | "
            f"val_adapted={val_meta:.5f} | val_tasks={val_task_n} | "
            f"step={meta_step_seconds:.2f}s | peak_alloc={peak_allocated_mb:.1f}MB"
        )

        if val_meta < best_val:
            best_val, best_step, bad_steps = val_meta, step, 0
            save_adapter(model, tokenizer, args.out, vars_to_json(args))
        else:
            bad_steps += 1
        if bad_steps >= args.patience:
            print(f"Early stopping at step {step}; best step={best_step}, val={best_val:.5f}")
            break

    pd.DataFrame(history).to_csv(args.out.parent / "stage1_history.csv", index=False)
    (args.out.parent / "stage1_summary.json").write_text(
        json.dumps({"best_step": best_step, "best_validation_loss": best_val,
                    "tasks": tasks, "trainable_parameter_names": trainable_names,
                    "elapsed_seconds": time.time()-start_time},
                   ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Best Stage 1 adapter: {args.out}")


def vars_to_json(args):
    return {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}

if __name__ == "__main__":
    main()
