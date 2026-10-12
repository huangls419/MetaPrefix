# stage1_sbs_maml.py
import os
import sys
import random
import logging
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import tools.patch_peft  # noqa: F401

import torch
import higher
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    get_polynomial_decay_schedule_with_warmup,
)

from peft import get_peft_model, PrefixTuningConfig, TaskType
from utils.set_seed import set_seed
from utils.log_helper import logger_init


class Stage1Config:
    def __init__(self):
        self.project_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_name_or_path = os.path.join(self.project_dir, "ProtGPT2")
        self.tokenizer_name_or_path = self.model_name_or_path

        self.train_path = os.path.join(self.project_dir, "data", "maml", "train_dataset.csv")
        self.test_path = os.path.join(self.project_dir, "data", "maml", "test_dataset.csv")

        self.num_virtual_tokens = 10
        self.encoder_hidden_size = 1280
        self.max_length = 50

        self.outer_lr = 1e-6
        self.inner_lr = 1e-4
        self.weight_decay = 0.01
        self.num_epochs = 5
        self.batch_size = 8
        self.min_samples_per_task = 2
        self.random_seed = 42

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model_save_dir = os.path.join(self.project_dir, "cache", "maml", "stage1")
        self.logs_save_dir = os.path.join(self.project_dir, "logs", "maml", "stage1")
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.logs_save_dir, exist_ok=True)

        self.writer = SummaryWriter(os.path.join(self.logs_save_dir, "tensorboard"))
        logger_init(
            log_file_name="stage1_sbs_maml",
            log_level=logging.INFO,
            log_dir=self.logs_save_dir,
        )

        self.peft_config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=self.num_virtual_tokens,
            encoder_hidden_size=self.encoder_hidden_size,
            prefix_projection=True,
        )


def build_task_loaders(config, tokenizer):
    dataset = load_dataset(
        "csv",
        data_files={
            "train": config.train_path,
            "test": config.test_path,
        },
    )

    task_columns = [c for c in dataset["train"].column_names if c != "Sequence"]

    def preprocess(examples):
        inputs = [str(x) for x in examples["Sequence"]]
        model_inputs = tokenizer(inputs, add_special_tokens=False)

        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        for i in range(len(inputs)):
            input_ids = [tokenizer.eos_token_id] + model_inputs["input_ids"][i] + [tokenizer.eos_token_id]

            if len(input_ids) < config.max_length:
                pad_len = config.max_length - len(input_ids)
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                attention_mask = [1] * (len(input_ids) - pad_len) + [0] * pad_len
                labels = input_ids.copy()
                labels[-pad_len:] = [-100] * pad_len
            else:
                input_ids = input_ids[:config.max_length]
                attention_mask = [1] * config.max_length
                labels = input_ids.copy()

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)

        return {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "labels": labels_list,
        }

    task_loaders = {}

    for task in task_columns:
        def is_positive(example):
            value = str(example.get(task, "")).strip().lower()
            return value in {"1", "1.0", "true", "yes", "positive"}

        train_ds = dataset["train"].filter(is_positive)
        test_ds = dataset["test"].filter(is_positive)

        if len(train_ds) < config.min_samples_per_task:
            logging.warning(f"Skip task {task}: train samples={len(train_ds)}")
            continue

        if len(test_ds) < 1:
            split = train_ds.train_test_split(test_size=0.2, seed=config.random_seed)
            support_ds, query_ds = split["train"], split["test"]
        else:
            support_ds, query_ds = train_ds, test_ds

        support_ds = support_ds.map(
            preprocess,
            batched=True,
            remove_columns=support_ds.column_names,
            load_from_cache_file=False,
        )
        query_ds = query_ds.map(
            preprocess,
            batched=True,
            remove_columns=query_ds.column_names,
            load_from_cache_file=False,
        )

        support_loader = DataLoader(
            support_ds,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=default_data_collator,
        )
        query_loader = DataLoader(
            query_ds,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=default_data_collator,
        )

        task_loaders[task] = (support_loader, query_loader)
        logging.info(
            f"Task {task}: support={len(support_ds)}, query={len(query_ds)}"
        )

    return task_loaders


class SBSMAMLTrainer:
    def __init__(self, model, config, total_steps):
        self.model = model
        self.config = config

        self.inner_params = [
            p for n, p in self.model.named_parameters()
            if "prompt_encoder" in n and p.requires_grad
        ]

        self.outer_optimizer = torch.optim.AdamW(
            self.inner_params,
            lr=config.outer_lr,
            weight_decay=config.weight_decay,
        )

        self.lr_scheduler = get_polynomial_decay_schedule_with_warmup(
            self.outer_optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps,
            lr_end=1e-10,
            power=3,
        )

    def inner_update(self, fmodel, diffopt, batch):
        loss = fmodel(**batch).loss
        diffopt.step(loss)
        return loss.item()

    def meta_update(self, task_batches):
        self.outer_optimizer.zero_grad()

        total_inner_loss = 0.0
        total_outer_loss = 0.0
        count = 0

        for support_batch, query_batch in task_batches:
            support_batch = {
                k: v.to(self.config.device) for k, v in support_batch.items()
            }
            query_batch = {
                k: v.to(self.config.device) for k, v in query_batch.items()
            }

            inner_opt = torch.optim.SGD(
                self.inner_params,
                lr=self.config.inner_lr,
            )

            with higher.innerloop_ctx(
                self.model,
                inner_opt,
                copy_initial_weights=False,
                track_higher_grads=True,
                override={"lr": [self.config.inner_lr]},
            ) as (fmodel, diffopt):
                inner_loss = self.inner_update(fmodel, diffopt, support_batch)
                query_loss = fmodel(**query_batch).loss

                # Key point: SBS-MAML accumulates meta-gradients serially, not stepping here
                query_loss.backward()

            total_inner_loss += inner_loss
            total_outer_loss += query_loss.item()
            count += 1

        self.outer_optimizer.step()
        self.lr_scheduler.step()

        return total_outer_loss / max(count, 1), total_inner_loss / max(count, 1)


def train(config):
    set_seed(config.random_seed)

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    task_loaders = build_task_loaders(config, tokenizer)
    logging.info(f"Valid tasks: {list(task_loaders.keys())}")

    model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path)
    model = get_peft_model(model, config.peft_config)
    model.print_trainable_parameters()
    model.to(config.device)

    total_steps = config.num_epochs * max(len(task_loaders), 1)
    trainer = SBSMAMLTrainer(model, config, total_steps)

    for epoch in range(config.num_epochs):
        tasks = list(task_loaders.keys())
        random.shuffle(tasks)

        task_batches = []
        for task in tasks:
            support_loader, query_loader = task_loaders[task]
            support_iter = iter(support_loader)
            query_iter = iter(query_loader)

            try:
                support_batch = next(support_iter)
                query_batch = next(query_iter)
            except StopIteration:
                continue

            task_batches.append((support_batch, query_batch))

        if not task_batches:
            continue

        outer_loss, inner_loss = trainer.meta_update(task_batches)

        config.writer.add_scalar("train/outer_loss", outer_loss, epoch)
        config.writer.add_scalar("train/inner_loss", inner_loss, epoch)

        logging.info(
            f"Epoch {epoch + 1}/{config.num_epochs} | "
            f"outer_loss={outer_loss:.4f} | inner_loss={inner_loss:.4f}"
        )

    save_path = os.path.join(
        config.model_save_dir,
        f"prefix_SBSMAML_E{config.num_epochs}_"
        f"LR{config.outer_lr}_BS{config.batch_size}_"
        f"ML{config.max_length}_VT{config.num_virtual_tokens}",
    )

    dummy_ids = torch.tensor([[tokenizer.eos_token_id]], device=config.device)
    hidden_states = model.base_model.transformer.wte(dummy_ids)

    try:
        model.save_pretrained(save_path, hidden_states=hidden_states)
    except TypeError:
        model.save_pretrained(save_path)
        torch.save(
            {"hidden_states": hidden_states},
            os.path.join(save_path, "hidden_states.pt"),
        )

    tokenizer.save_pretrained(save_path)
    logging.info(f"Saved Stage1 model to {save_path}")


if __name__ == "__main__":
    cfg = Stage1Config()
    train(cfg)