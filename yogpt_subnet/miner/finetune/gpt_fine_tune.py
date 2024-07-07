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
from huggingface_hub import HfApi, Repository, create_repo, login
from torch.utils.data import (DataLoader, Dataset, RandomSampler,
                              SequentialSampler, random_split)
from transformers import (AdamW, GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          get_linear_schedule_with_warmup)

from yogpt_subnet.miner.utils.helpers import update_job_status

nltk.download('punkt')

class GPT2Dataset(Dataset):
    def __init__(self, txt_list, tokenizer, max_length=768):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []

        for txt in txt_list:
            encodings_dict = tokenizer(txt, truncation=True, max_length=max_length, padding="max_length")
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

async def fine_tune_gpt(base_model, dataset_id, new_model_name, hf_token, job_id):
    """Fine-tune GPT-2 model and upload it to Hugging Face."""
    base_model = str(base_model)
    print("------basemodel specified-----" + base_model)
    print(".......new_model_name ........" + new_model_name)
    print(".......dataset specified ........" + dataset_id)

    # Designate directories
    dataset_dir = os.path.join("data", dataset_id)
    os.makedirs(dataset_dir, exist_ok=True)

    try:
        # Load dataset
        dataset = load_dataset(dataset_id, split="train", cache_dir=dataset_dir)

        # Load GPT-2 tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained(base_model, pad_token="<|endoftext|>")

        # Load GPT-2 model
        model = GPT2LMHeadModel.from_pretrained(base_model)

        # Set training parameters
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=4,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            save_strategy='epoch',
            evaluation_strategy='steps',
            eval_steps=500,
            logging_steps=500,
            load_best_model_at_end=True
        )

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer
        )

        # Train model
        trainer.train()

        # Create repository on Hugging Face and clone it locally
        api = HfApi()
        repo_url = api.create_repo(repo_id=job_id, token=hf_token)
        repo = Repository(local_dir=f"models/{job_id}", clone_from=repo_url, token=hf_token)

        model_to_save.save_pretrained(repo.local_dir)
        tokenizer.save_pretrained(repo.local_dir)

        repo.git_add(pattern=".")
        repo.git_commit("Add fine-tuned model files")
        repo.git_push()

        return repo_url, avg_val_loss, avg_val_accuracy

    except Exception as e:
        await update_job_status(job_id, 'pending')
        raise RuntimeError(f"Training pipeline encountered an error: {str(e)}")

    finally:
        # Clean up the dataset directory
        shutil.rmtree(dataset_dir)
