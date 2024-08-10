import datetime
import os
import random
import shutil
import time

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import wandb
from datasets import load_dataset
from huggingface_hub import HfApi, Repository, create_repo, login
from transformers import (DataCollatorForLanguageModeling, GPT2LMHeadModel,
                          GPT2Tokenizer, Trainer, TrainingArguments)

from yogpt_subnet.miner.models.storage.hugging_face_store import \
    HuggingFaceModelStore
from yogpt_subnet.miner.utils.helpers import update_job_status


def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

async def fine_tune_gpt(base_model, dataset_id, new_model_name, hf_token, job_id):
    """Fine-tune GPT-2 model and upload it to Hugging Face."""
    print("Starting fine-tuning process...")
    base_model = str(base_model)
    print("*------base model specified-----*" + base_model)
    print("*.......new model name ........*" + new_model_name)
    print("*.......dataset specified ........*" + dataset_id)

    pipeline_start_time = time.time()
    dataset_dir = os.path.join("data", dataset_id)
    os.makedirs(dataset_dir, exist_ok=True)

    try:
        # Set the WANDB API key
        os.environ["WANDB_API_KEY"] = "efa7d98857a922cbe11e78fa1ac22b62a414fbf3"
        
        # Login to Hugging Face
        login(hf_token)

        # Initialize wandb
        wandb.init(project=job_id, entity="ai-research-lab", config={
            "base_model": base_model,
            "new_model_name": new_model_name,
            "dataset_id": dataset_id,
        })

        dataset = load_dataset(dataset_id, split="train", cache_dir=dataset_dir, trust_remote_code=True)

        # Load GPT-2 tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained(base_model, pad_token="")

        # Load GPT-2 model
        model = GPT2LMHeadModel.from_pretrained(base_model)

        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True)

        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

        def compute_loss(model, inputs):
            labels = inputs.pop("labels")
            outputs = model(**inputs, labels=labels)
            return outputs.loss

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=10,
            per_device_train_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            label_names=['input_ids', 'attention_mask'],
            logging_dir='./logs',
            eval_steps=500,
            logging_steps=500,
            run_name="gpt2_model_running",
            fp16=False,
            report_to="wandb"
        )

        print("Setting up trainer........")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_loss,
            data_collator=data_collator
        )
        print("Trainer set up successfully.")

        print("Starting training...")

        # Train model
        try:
            train_result = trainer.train()
            train_loss = train_result.training_loss
            accuracy = 0

        except Exception as e:
            await update_job_status(job_id, 'pending')
            raise RuntimeError(f"Training failed: {str(e)}")

        store = HuggingFaceModelStore()
        repo_url = store.upload_model(model, tokenizer, job_id)

        # Capture the end time
        pipeline_end_time = time.time()
        total_pipeline_time = format_time(pipeline_end_time - pipeline_start_time)
        print("........ model details...........")
        print(repo_url)
        print(train_loss)
        print(accuracy)
        print(total_pipeline_time)
        return repo_url, train_loss, accuracy, total_pipeline_time

    except Exception as e:
        await update_job_status(job_id, 'pending')
        return None, None, None, None

    finally:
        try:
            print("Cleaning up dataset directory...")
            shutil.rmtree(dataset_dir)
            print("Cleanup completed.")
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

        # Ensure the job status is updated to pending if the model was not saved to wandb
        if 'repo_url' not in locals():
            await update_job_status(job_id, 'pending')
