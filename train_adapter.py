import os
import torch
import random
import numpy as np

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from TrainableHuggingfaceChatbot import TrainableHuggingfaceChatbot
from load_dataset_CV import *


# =========================
# CONFIG
# =========================

MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
DOMAIN = "GDPR"

OUTPUT_ROOT = "./adapters"

MAX_LENGTH = 1024
BATCH_SIZE = 1
GRAD_ACCUM = 4
LR = 1e-4
EPOCHS = 2
SEED = 42


NORM2CHOICE = {
    "PROHIBITED": "A. Prohibited",
    "PERMITTED": "B. Permitted",
    "NOT_RELATED": "C. Not related",
}


# =========================
# UTILS
# =========================

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_ci_decision_example(row):
    prompt = f"""You are an expert in Contextual Integrity and {DOMAIN} regulations.

Given the following event, first analyze it using the elements of Contextual Integrity,
then determine whether the event is allowed under {DOMAIN}.

Event:
{row['case_content']}

Follow this format strictly.

Contextual Integrity Analysis:
- Sender:
- Recipient:
- Subject:
- Information Type:
- Transmission Principle:

Decision:
Choice: [A. Prohibited | B. Permitted | C. Not related]

Assistant:
"""

    target = f"""Contextual Integrity Analysis:
- Sender: {row.get('sender', '')}
- Recipient: {row.get('recipient', '')}
- Subject: {row.get('subject', '')}
- Information Type: {row.get('information_type', '')}
- Transmission Principle: {row.get('purpose', '')}

Decision:
{NORM2CHOICE[row['norm_type']]}
"""

    return {
        "prompt": prompt,
        "target": target,
        "text": prompt + target
    }


# =========================
# TOKENIZATION WITH MASKING
# =========================

def build_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_with_masking(example, tokenizer):
    full = tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LENGTH,
    )

    prompt_ids = tokenizer(
        example["prompt"],
        truncation=True,
        max_length=MAX_LENGTH,
    )["input_ids"]

    labels = full["input_ids"].copy()
    labels[: len(prompt_ids)] = [-100] * len(prompt_ids)

    full["labels"] = labels
    return full


# =========================
# MAIN TRAIN LOOP
# =========================

def train_one_fold(fold_id, train_split):
    print(f"\n🚀 Training fold {fold_id}")

    adapter_dir = os.path.join(OUTPUT_ROOT, f"fold_{fold_id}")
    os.makedirs(adapter_dir, exist_ok=True)

    # build dataset
    examples = [build_ci_decision_example(row) for row in train_split]
    dataset = Dataset.from_list(examples)

    tokenizer = build_tokenizer(MODEL_NAME)

    tokenized_dataset = dataset.map(
        lambda x: tokenize_with_masking(x, tokenizer),
        remove_columns=dataset.column_names,
    )

    chatbot = TrainableHuggingfaceChatbot(
        model=MODEL_NAME,
        lora_r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=("q_proj", "v_proj"),
    )

    model = chatbot.model
    model.train()

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=adapter_dir,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        fp16=True,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        optim="adamw_torch",
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    print(f"✅ Fold {fold_id} completed. Adapter saved to {adapter_dir}")


# =========================
# ENTRY POINT
# =========================

def main():
    set_seeds(SEED)

    k_fold_data = load_k_fold_dataset()

    for fold_id, domains in k_fold_data.items():
        train_split = domains[DOMAIN]["train"]
        train_one_fold(fold_id, train_split)


if __name__ == "__main__":
    main()
