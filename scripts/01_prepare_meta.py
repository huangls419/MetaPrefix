# -*- coding: utf-8 -*-
"""Prepare group-disjoint Support/Meta-query sets for Stage 1 and Stage 2."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from common import (ROOT, DEFAULT_OUT, CANCER_COLS, ZERO_SHOT_TASKS,
                    read_csv, check_partition_integrity)


def choose_query_groups(df: pd.DataFrame, tasks: list[str], fraction: float,
                        restarts: int, seed: int):
    grouped = df.groupby("split_group_id")[tasks].max().astype(int)
    gids = grouped.index.to_numpy(dtype=str)
    y = grouped.to_numpy(dtype=float)
    n_groups, n_tasks = y.shape
    if n_groups < 2:
        raise ValueError("Train has fewer than two split groups; cannot create meta Support/Query.")
    n_query = max(1, int(round(n_groups * fraction)))
    n_query = min(n_query, n_groups - 1)
    totals = y.sum(axis=0)
    # Balance all cancer tasks, with strong penalties for losing task coverage.
    eligible = totals >= 2
    rng = np.random.default_rng(seed)
    best_score, best_idx = float("inf"), None
    for _ in range(restarts):
        idx = rng.choice(n_groups, size=n_query, replace=False)
        q_counts = y[idx].sum(axis=0)
        s_counts = totals - q_counts
        rel_err = np.abs(q_counts - fraction * totals) / np.maximum(totals, 1)
        active = totals > 0
        score = float(rel_err[active].mean()) if active.any() else 0.0
        missing = eligible & ((q_counts == 0) | (s_counts == 0))
        score += 2.0 * float(missing.sum()) / max(int(eligible.sum()), 1)
        # Also target the global group ratio.
        score += abs(len(idx) / n_groups - fraction)
        if score < best_score:
            best_score, best_idx = score, idx.copy()
    query_groups = set(gids[best_idx].tolist())
    return query_groups, grouped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--query-fraction", type=float, default=0.20)
    ap.add_argument("--min-groups", type=int, default=5)
    ap.add_argument("--restarts", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if not 0.05 <= args.query_fraction <= 0.5:
        raise ValueError("query-fraction should be between 0.05 and 0.5")

    train = read_csv(args.out / "train.csv")
    valid = read_csv(args.out / "valid.csv")
    test = read_csv(args.out / "test.csv")
    zero_path, buf_path = args.out / "zero_shot.csv", args.out / "zs_buffer.csv"
    zero = read_csv(zero_path) if zero_path.exists() else None
    buffer = read_csv(buf_path) if buf_path.exists() else None
    check_partition_integrity(train, valid, test, zero, buffer)

    tasks = [c for c in CANCER_COLS if c not in ZERO_SHOT_TASKS]
    for task in ZERO_SHOT_TASKS:
        if int(train[task].sum()) or int(valid[task].sum()) or int(test[task].sum()):
            raise RuntimeError(f"Zero-shot target {task} must not appear in Train/Validation/Test.")

    # Task regime is calculated from the full curated partition collection,
    # counting original CD-HIT clusters once per cancer label.
    all_parts = [train, valid, test]
    if zero is not None:
        all_parts.append(zero)
    if buffer is not None:
        all_parts.append(buffer)
    all_data = pd.concat(all_parts, ignore_index=True).drop_duplicates("peptide_id")
    total_cluster_counts = {
        task: int(all_data.loc[all_data[task] == 1, "cluster_id"].nunique())
        for task in CANCER_COLS
    }

    query_groups, group_matrix = choose_query_groups(
        train, tasks, args.query_fraction, args.restarts, args.seed
    )
    train = train.copy()
    train["meta_role"] = np.where(
        train["split_group_id"].astype(str).isin(query_groups),
        "meta_query", "meta_support"
    )
    support = train[train.meta_role == "meta_support"].drop(columns="meta_role")
    query = train[train.meta_role == "meta_query"].drop(columns="meta_role")

    overlap = set(support.split_group_id) & set(query.split_group_id)
    if overlap:
        raise RuntimeError(f"Internal split leakage: {len(overlap)} groups overlap.")
    if set(support.Sequence) & set(query.Sequence):
        raise RuntimeError("Exact sequence overlap between meta Support and Query.")

    args.out.mkdir(parents=True, exist_ok=True)
    support.to_csv(args.out / "meta_support.csv", index=False)
    query.to_csv(args.out / "meta_query.csv", index=False)
    train[["peptide_id", "Sequence", "cluster_id", "split_group_id", "meta_role"]].to_csv(
        args.out / "meta_roles.csv", index=False
    )

    rows = []
    include_tasks = []
    for task in tasks:
        p_all = train[train[task] == 1]
        p_sup = support[support[task] == 1]
        p_qry = query[query[task] == 1]
        n_groups = p_all.split_group_id.nunique()
        n_sup_groups = p_sup.split_group_id.nunique()
        n_qry_groups = p_qry.split_group_id.nunique()
        total_clusters = total_cluster_counts.get(task, 0)
        if task in ZERO_SHOT_TASKS:
            regime = "zero-shot held-out"
            status = "holdout"
            reason = "predefined held-out task; excluded from Stage 1 and Stage 2 adaptation"
        elif total_clusters > 100:
            regime = "major-shot"
            status = "include" if (
                n_groups >= args.min_groups and n_sup_groups >= 2 and n_qry_groups >= 1
            ) else "skip"
            reason = "" if status == "include" else (
                f"insufficient independent groups (total={n_groups}, support={n_sup_groups}, query={n_qry_groups})"
            )
        elif total_clusters >= 10:
            regime = "few-shot"
            status = "include" if (
                n_groups >= args.min_groups and n_sup_groups >= 2 and n_qry_groups >= 1
            ) else "skip"
            reason = "" if status == "include" else (
                f"insufficient independent groups (total={n_groups}, support={n_sup_groups}, query={n_qry_groups})"
            )
        elif total_clusters >= 1:
            # Do not silently train a 1-9-cluster task as if it were few-shot.
            # It must be explicitly reclassified/held out in a separate protocol.
            regime = "low-support (1-9 clusters; excluded from Stage 1)"
            status = "skip"
            reason = "below the predefined few-shot threshold; explicitly define a separate held-out protocol if evaluated"
        else:
            regime = "no positive annotation"
            status = "skip"
            reason = "no curated positive sequences for this cancer task"
        rows.append({
            "Cancer": task,
            "Task_regime": regime,
            "Total_positive_CD-HIT_clusters": total_cluster_counts.get(task, 0),
            "Train_positive_sequences": len(p_all),
            "Train_positive_groups": n_groups,
            "Support_positive_sequences": len(p_sup),
            "Support_positive_groups": n_sup_groups,
            "MetaQuery_positive_sequences": len(p_qry),
            "MetaQuery_positive_groups": n_qry_groups,
            "Validation_positive_sequences": int(valid[task].sum()),
            "Validation_positive_groups": int(valid.loc[valid[task] == 1, "split_group_id"].nunique()),
            "Test_positive_sequences": int(test[task].sum()),
            "Test_positive_groups": int(test.loc[test[task] == 1, "split_group_id"].nunique()),
            "Stage1_status": status,
            "Reason": reason,
        })
        if status == "include":
            include_tasks.append(task)

    # Explicit rows for the six held-out Zero-shot cancer tasks.
    for task in ZERO_SHOT_TASKS:
        z = zero[zero[task] == 1] if zero is not None else pd.DataFrame()
        rows.append({
            "Cancer": task,
            "Task_regime": "zero-shot held-out",
            "Total_positive_CD-HIT_clusters": total_cluster_counts.get(task, 0),
            "Train_positive_sequences": int(train[task].sum()),
            "Train_positive_groups": 0,
            "Support_positive_sequences": 0,
            "Support_positive_groups": 0,
            "MetaQuery_positive_sequences": 0,
            "MetaQuery_positive_groups": 0,
            "Validation_positive_sequences": int(valid[task].sum()),
            "Validation_positive_groups": int(valid.loc[valid[task] == 1, "split_group_id"].nunique()),
            "Test_positive_sequences": int(test[task].sum()),
            "Test_positive_groups": int(test.loc[test[task] == 1, "split_group_id"].nunique()),
            "Zero_shot_positive_sequences": int(z[task].sum()) if len(z) else 0,
            "Stage1_status": "holdout",
            "Reason": "predefined held-out task; excluded from Stage 1 and Stage 2 adaptation",
        })

    manifest = pd.DataFrame(rows)
    manifest.to_csv(args.out / "meta_task_manifest.csv", index=False)
    print(f"Train sequences: {len(train):,}; split groups: {train.split_group_id.nunique():,}")
    print(f"Meta-support: {len(support):,} sequences / {support.split_group_id.nunique():,} groups")
    print(f"Meta-query:   {len(query):,} sequences / {query.split_group_id.nunique():,} groups")
    print(f"Stage 1 tasks included: {len(include_tasks)} / {len(tasks)}")
    print(manifest.to_string(index=False))
    if not include_tasks:
        raise RuntimeError("No Stage 1 tasks passed min-groups criteria. Review meta_task_manifest.csv.")

if __name__ == "__main__":
    main()
