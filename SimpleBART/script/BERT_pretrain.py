import argparse
import logging
import os
import math
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForMaskedLM,
                          DataCollatorForLanguageModeling, Trainer, TrainingArguments)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune BERT for Japanese MLM task")
    parser.add_argument("--train_file", type=str, default="../data/sample_mask_train.csv", help="Path to the training CSV file.")
    parser.add_argument("--validation_file", type=str, default="../data/sample_mask_valid.csv", help="Path to the validation CSV file.")
    parser.add_argument("--model_name_or_path", type=str, default="tohoku-nlp/bert-base-japanese-whole-word-masking",
                        help="Pretrained model name or path.")
    parser.add_argument("--output_dir", type=str, default="./../SNOW_mlm_output", help="Where to save the model.")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Maximum sequence length.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=32, help="Batch size per device for training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32, help="Batch size per device for evaluation.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Number of training epochs.")
    return parser.parse_args()

def main():
    args = parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path)

    logger.info("Loading dataset...")
    data_files = {"train": args.train_file, "validation": args.validation_file}
    datasets = load_dataset("csv", data_files=data_files)

    # Check dataset columns and use correct column names for train and validation
    train_columns = datasets["train"].column_names
    logger.info(f"Training dataset columns: {train_columns}")
    text_column_train = "text" if "text" in train_columns else train_columns[0]  # Use first column if "text" is not found
    label_column_train = "label" if "label" in train_columns else train_columns[1]  # Assume second column for labels

    validation_columns = datasets["validation"].column_names
    logger.info(f"Validation dataset columns: {validation_columns}")
    text_column_validation = "text" if "text" in validation_columns else validation_columns[0]
    label_column_validation = "label" if "label" in validation_columns else validation_columns[1]

    def tokenize_function_train(examples):
        try:
            inputs = tokenizer(examples[text_column_train], padding="max_length", truncation=True, max_length=args.max_seq_length)
            labels = tokenizer(examples[label_column_train], padding="max_length", truncation=True, max_length=args.max_seq_length)
            inputs["labels"] = labels["input_ids"]
            return inputs
        except Exception as e:
            logger.error(f"Error tokenizing training examples: {examples}")
            raise e

    def tokenize_function_validation(examples):
        try:
            inputs = tokenizer(examples[text_column_validation], padding="max_length", truncation=True, max_length=args.max_seq_length)
            labels = tokenizer(examples[label_column_validation], padding="max_length", truncation=True, max_length=args.max_seq_length)
            inputs["labels"] = labels["input_ids"]
            return inputs
        except Exception as e:
            logger.error(f"Error tokenizing validation examples: {examples}")
            raise e

    logger.info("Tokenizing training dataset...")
    tokenized_train_dataset = datasets["train"].map(tokenize_function_train, batched=True, remove_columns=[text_column_train, label_column_train])

    logger.info("Tokenizing validation dataset...")
    tokenized_validation_dataset = datasets["validation"].map(tokenize_function_validation, batched=True, remove_columns=[text_column_validation, label_column_validation])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        save_total_limit=2,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=50,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_validation_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
