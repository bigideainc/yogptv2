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
from datasets import load_dataset
from huggingface_hub import HfApi, Repository, create_repo, login
from transformers import (GPT2LMHeadModel, GPT2Tokenizer, Trainer,TrainingArguments, DataCollatorForLanguageModeling)
from yogpt_subnet.miner.models.storage.hugging_face_store import \
    HuggingFaceModelStore
from yogpt_subnet.miner.utils.helpers import update_job_status



def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

async def fine_tune_gpt(base_model, dataset_id, new_model_name, hf_token, job_id):
    """Fine-tune GPT-2 model and upload it to Hugging Face."""
    base_model = str(base_model)
    print("------base model specified-----" + base_model)
    print(".......new model name ........" + new_model_name)
    print(".......dataset specified ........" + dataset_id)

    # Capture the start time
    pipeline_start_time = time.time()

    # Designate directories
    dataset_dir = os.path.join("data", dataset_id)
    os.makedirs(dataset_dir, exist_ok=True)

    try:
        # Login to Hugging Face
        login(hf_token)
        dataset = load_dataset(dataset_id, split="train", cache_dir=dataset_dir,trust_remote_code=True)

        # Load GPT-2 tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained(base_model, pad_token="<|endoftext|>")

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
            num_train_epochs=1,
            per_device_train_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            label_names=['input_ids', 'attention_mask'],
            logging_dir='./logs',
            eval_steps=500,
            logging_steps=500,
            run_name="gpt2_model_running",
            fp16=False,

        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_loss,
            data_collator=data_collator
        )
        
        train_result = trainer.train()
        train_loss = train_result.training_loss
        accuracy = 0
        store = HuggingFaceModelStore()
        repo_url = store.upload_model(model, tokenizer, job_id)
        
        # Capture the end time
        pipeline_end_time = time.time()
        total_pipeline_time = format_time(pipeline_end_time - pipeline_start_time)
        pipeline_end_time = time.time()
        total_pipeline_time = format_time(pipeline_end_time - pipeline_start_time)
        print("........ model details...........")
        print(repo_url)
        print(train_loss)
        print(accuracy)
        print(total_pipeline_time)
        return repo_url, train_loss,accuracy, total_pipeline_time

    except Exception as e:
        await update_job_status(job_id, 'pending')
        return None, None, None, None

    finally:
        # Clean up the dataset directory
        shutil.rmtree(dataset_dir)
