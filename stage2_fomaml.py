# stage2_fomaml.py
import os
import sys
import logging

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

from peft import PeftModel
from utils.set_seed import set_seed
from utils.log_helper import logger_init


class Stage2Config:
    def __init__(self):
        self.project_dir = os.path.dirname(os.path.abspath(__file__))

        # Change to the path output by Stage1
        self.stage1_model_path = os.path.join(
            self.project_dir,
            "cache/maml/stage1/prefix_SBSMAML_E5_LR1e-06_BS8_ML50_VT10",
        )

        self.target_cancer = "lung"
        self.target_column = self.target_cancer

        self.train_path = os.path.join(self.project_dir, "data", "maml", "train_dataset.csv")
        self.test_path = os.path.join(self.project_dir, "data", "maml", "test_dataset.csv")

        self.support_ratio = 0.8
        self.inner_lr = 1e-4
        self.outer_lr = 5e-5
        self.weight_decay = 0.01
        self.num_epochs = 200
        self.batch_size = 1
        self.max_length = 50
        self.random_seed = 42

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model_save_dir = os.path.join(
            self.project_dir, "cache", "maml", "stage2", self.target_cancer
        )
        self.logs_save_dir = os.path.join(self.project_dir, "logs", "maml", "stage2")
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.logs_save_dir, exist_ok=True)

        self.writer = SummaryWriter(os.path.join(self.logs_save_dir, "tensorboard"))
        logger_init(
            log_file_name=f"{self.target_cancer}_stage2_fomaml",
            log_level=logging.INFO,
            log_dir=self.logs_save_dir,
        )


def load_stage2_data(config, tokenizer):
    train_ds = load_dataset("csv", data_files={"train": config.train_path})["train"]
    test_ds = load_dataset("csv", data_files={"test": config.test_path})["test"]

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def is_positive(example):
        value = str(example.get(config.target_column, "")).strip().lower()
        return value in {"1", "1.0", "true", "yes", "positive"}

    train_ds = train_ds.filter(is_positive)
    test_ds = test_ds.filter(is_positive)

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

    train_ds = train_ds.map(
        preprocess,
        batched=True,
        remove_columns=train_ds.column_names,
        load_from_cache_file=False,
    )
    test_ds = test_ds.map(
        preprocess,
        batched=True,
        remove_columns=test_ds.column_names,
        load_from_cache_file=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=default_data_collator,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=default_data_collator,
    )

    return train_loader, test_loader


class Stage2FOMAMLTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.best_loss = float("inf")

    def outer_update(self, support_batch, query_batch):
        inner_params = [
            p for n, p in self.model.named_parameters()
            if "prompt_encoder" in n and p.requires_grad
        ]

        inner_opt = torch.optim.SGD(inner_params, lr=self.config.inner_lr)

        with higher.innerloop_ctx(
            self.model,
            inner_opt,
            copy_initial_weights=False,
            track_higher_grads=True,
            override={"lr": [self.config.inner_lr]},
        ) as (fmodel, diffopt):
            inner_loss = fmodel(**support_batch).loss
            diffopt.step(inner_loss)
            outer_loss = fmodel(**query_batch).loss

        return outer_loss, inner_loss.item()

    def evaluate(self, test_loader):
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(self.config.device) for k, v in batch.items()}
                loss = self.model(**batch).loss
                total_loss += loss.item() * batch["input_ids"].size(0)
                total_samples += batch["input_ids"].size(0)

        return total_loss / max(total_samples, 1)

    def save_model(self, epoch, tokenizer):
        save_path = os.path.join(
            self.config.model_save_dir,
            f"{self.config.target_cancer}_epoch{epoch + 1}_loss{self.best_loss:.4f}",
        )
        os.makedirs(save_path, exist_ok=True)

        dummy_ids = torch.tensor([[tokenizer.eos_token_id]], device=self.config.device)
        hidden_states = self.model.base_model.transformer.wte(dummy_ids)

        try:
            self.model.save_pretrained(save_path, hidden_states=hidden_states)
        except TypeError:
            self.model.save_pretrained(save_path)
            torch.save(
                {"hidden_states": hidden_states},
                os.path.join(save_path, "hidden_states.pt"),
            )

        tokenizer.save_pretrained(save_path)
        logging.info(f"Saved Stage2 model to {save_path}")

    def train(self, train_loader, test_loader, tokenizer):
        self.model.to(self.config.device)

        batches = list(train_loader)
        support_num = int(len(batches) * self.config.support_ratio)
        support_batches = batches[:support_num]
        query_batches = batches[support_num:]

        if len(query_batches) == 0:
            query_batches = support_batches[-1:]

        outer_params = [
            p for n, p in self.model.named_parameters()
            if "prompt_encoder" in n and p.requires_grad
        ]

        outer_optimizer = torch.optim.AdamW(
            outer_params,
            lr=self.config.outer_lr,
            weight_decay=self.config.weight_decay,
        )

        total_steps = self.config.num_epochs * max(
            min(len(support_batches), len(query_batches)), 1
        )

        lr_scheduler = get_polynomial_decay_schedule_with_warmup(
            outer_optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps,
            lr_end=1e-10,
            power=3,
        )

        for epoch in range(self.config.num_epochs):
            total_outer = 0.0
            total_inner = 0.0
            count = 0

            for i in range(min(len(support_batches), len(query_batches))):
                support_batch = {
                    k: v.to(self.config.device)
                    for k, v in support_batches[i].items()
                }
                query_batch = {
                    k: v.to(self.config.device)
                    for k, v in query_batches[i].items()
                }

                outer_loss, inner_loss = self.outer_update(support_batch, query_batch)

                outer_optimizer.zero_grad()
                outer_loss.backward()
                outer_optimizer.step()
                lr_scheduler.step()

                total_outer += outer_loss.item()
                total_inner += inner_loss
                count += 1

            avg_outer = total_outer / max(count, 1)
            avg_inner = total_inner / max(count, 1)

            self.config.writer.add_scalar("train/outer_loss", avg_outer, epoch)
            self.config.writer.add_scalar("train/inner_loss", avg_inner, epoch)

            logging.info(
                f"Epoch {epoch + 1}/{self.config.num_epochs} | "
                f"outer_loss={avg_outer:.4f} | inner_loss={avg_inner:.4f}"
            )

            test_loss = self.evaluate(test_loader)
            logging.info(f"Test loss: {test_loss:.4f}")

            if avg_outer < self.best_loss:
                self.best_loss = avg_outer
                self.save_model(epoch, tokenizer)


def main():
    set_seed(42)

    config = Stage2Config()

    base_model = AutoModelForCausalLM.from_pretrained(
        os.path.join(config.project_dir, "ProtGPT2")
    )
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(config.project_dir, "ProtGPT2")
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = PeftModel.from_pretrained(base_model, config.stage1_model_path)

    train_loader, test_loader = load_stage2_data(config, tokenizer)

    trainer = Stage2FOMAMLTrainer(model, config)
    trainer.train(train_loader, test_loader, tokenizer)


if __name__ == "__main__":
    main()