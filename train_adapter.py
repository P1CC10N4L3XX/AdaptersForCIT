from transformers import Trainer, TrainingArguments
from TrainableHuggingfaceChatbot import TrainableHuggingfaceChatbot

MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
CSV_PATH = "/GDPR.csv"

OUTPUT_DIR = "./outputs"
ADAPTER_DIR = "./gdpr_lora_adapter"

MAX_LENGTH = 1024
BATCH_SIZE = 2
GRAD_ACCUM = 8
LR = 2e-4
EPOCHS = 3

def main():

    # ---- Dataset ----
    df = pd.read_csv(CSV_PATH)
    df = df.fillna("")

    def build_example(row):
        prompt = f"""You are an expert in GDPR and Contextual Integrity.

Given the following scenario, identify:
1. The GDPR norm type involved
2. The Contextual Integrity elements:
   - sender
   - recipient
   - subject
   - information_type

Scenario:
{row['case_content']}

Answer:
"""
        target = {
            "norm_type": row["norm_type"],
            "contextual_integrity": {
                "sender": row.get("sender", ""),
                "recipient": row.get("recipient", ""),
                "subject": row.get("subject", ""),
                "information_type": row.get("information_type", ""),
            },
        }

        return {
            "text": prompt + json.dumps(target, ensure_ascii=False)
        }

    examples = [build_example(row) for _, row in df.iterrows()]
    dataset = Dataset.from_list(examples)

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )

    tokenized_dataset = dataset.map(
        tokenize,
        remove_columns=["text"],
    )

    # ---- Modello + LoRA ----
    chatbot = TrainableHuggingfaceChatbot(
        model=MODEL_NAME,
        lora_r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=("q_proj", "v_proj"),
    )

    model = chatbot.model
    model.train()

    # ---- Data collator ----
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # ---- Training args ----
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        fp16=True,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        evaluation_strategy="no",
        report_to="none",
        optim="adamw_torch",
    )

    # ---- Trainer ----
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # ---- Training ----
    trainer.train()

    # ---- Salvataggio adapter ----
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)

    print("✅ Training completato. Adapter salvato in:", ADAPTER_DIR)


if __name__ == "__main__":
    main()
    

