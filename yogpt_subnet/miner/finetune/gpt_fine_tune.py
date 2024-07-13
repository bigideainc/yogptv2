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

        # Load dataset from Hugging Face
        dataset = load_dataset(dataset_id, split="train",use_auth_token=hf_token)

        # Split dataset into training and validation sets (90% train, 10% validation)
        split_dataset = dataset.train_test_split(test_size=0.1)
        train_dataset = split_dataset['train']
        eval_dataset = split_dataset['test']

        texts = train_dataset['text']  # Adjust as needed based on the dataset structure

        # Create custom dataset
        dataset = GPT2Dataset(texts, GPT2Tokenizer.from_pretrained(base_model), max_length=768)

        # DataLoader for training and validation
        batch_size = 16
        train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
        validation_dataloader = DataLoader(eval_dataset, sampler=SequentialSampler(eval_dataset), batch_size=batch_size)

        configuration = GPT2Config.from_pretrained(base_model, output_hidden_states=False)
        model = GPT2LMHeadModel.from_pretrained(base_model, config=configuration)
        tokenizer = GPT2Tokenizer.from_pretrained(base_model)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))
        tokenizer.padding_side = "left" 
        tokenizer.pad_token = tokenizer.eos_token
        model.resize_token_embeddings(len(tokenizer))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Training parameters
        epochs = 5
        learning_rate = 5e-4
        warmup_steps = 1e2
        epsilon = 1e-8
        sample_every = 100

        optimizer = AdamW(model.parameters(), lr=learning_rate, eps=epsilon)
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        total_t0 = time.time()
        training_stats = []

        for epoch_i in range(epochs):
            print(f"======== Epoch {epoch_i + 1} / {epochs} ========")
            print("Training...")

            t0 = time.time()
            total_train_loss = 0
            total_train_accuracy = 0
            model.train()

            for step, batch in enumerate(train_dataloader):
                b_input_ids = batch[0].to(device)
                b_labels = batch[0].to(device)
                b_masks = batch[1].to(device)

                model.zero_grad()

                outputs = model(b_input_ids, labels=b_labels, attention_mask=b_masks, token_type_ids=None)
                loss = outputs[0]
                logits = outputs[1]
                batch_loss = loss.item()
                total_train_loss += batch_loss

                preds = torch.argmax(logits, dim=-1)
                accuracy = (preds == b_labels).float().mean()
                total_train_accuracy += accuracy.item()

                if step % sample_every == 0 and not step == 0:
                    elapsed = format_time(time.time() - t0)
                    print(f"  Batch {step}  of  {len(train_dataloader)}. Loss: {batch_loss}. Accuracy: {accuracy.item()}.   Elapsed: {elapsed}.")

                    model.eval()
                    sample_outputs = model.generate(bos_token_id=random.randint(1, 30000), do_sample=True, top_k=50, max_length=200, top_p=0.95, num_return_sequences=1)
                    for i, sample_output in enumerate(sample_outputs):
                        print(f"{i}: {tokenizer.decode(sample_output, skip_special_tokens=True)}")

                    model.train()

                loss.backward()
                optimizer.step()
                scheduler.step()

            avg_train_loss = total_train_loss / len(train_dataloader)
            avg_train_accuracy = total_train_accuracy / len(train_dataloader)
            training_time = format_time(time.time() - t0)

            print(f"  Average training loss: {avg_train_loss:.2f}")
            print(f"  Average training accuracy: {avg_train_accuracy:.2f}")
            print(f"  Training epoch took: {training_time}")

            print("Running Validation...")

            t0 = time.time()
            model.eval()
            total_eval_loss = 0
            total_eval_accuracy = 0

            for batch in validation_dataloader:
                b_input_ids = batch[0].to(device)
                b_labels = batch[0].to(device)
                b_masks = batch[1].to(device)

                with torch.no_grad():
                    outputs = model(b_input_ids, attention_mask=b_masks, labels=b_labels)
                    loss = outputs[0]
                    logits = outputs[1]

                total_eval_loss += loss.item()

                preds = torch.argmax(logits, dim=-1)
                accuracy = (preds == b_labels).float().mean()
                total_eval_accuracy += accuracy.item()

            avg_val_loss = total_eval_loss / len(validation_dataloader)
            avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
            validation_time = format_time(time.time() - t0)

            print(f"  Validation Loss: {avg_val_loss:.2f}")
            print(f"  Validation Accuracy: {avg_val_accuracy:.2f}")
            print(f"  Validation took: {validation_time}")

            training_stats.append({
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Training Accuracy': avg_train_accuracy,
                'Valid. Accuracy': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            })

        print("Training complete!")
        print(f"Total training took {format_time(time.time() - total_t0)} (h:mm:ss)")

        df_stats = pd.DataFrame(data=training_stats).set_index('epoch')

        sns.set(style='darkgrid')
        sns.set(font_scale=1.5)
        plt.rcParams["figure.figsize"] = (12, 6)

        plt.plot(df_stats['Training Loss'], 'b-o', label="Training Loss")
        plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation Loss")
        plt.plot(df_stats['Training Accuracy'], 'r-o', label="Training Accuracy")
        plt.plot(df_stats['Valid. Accuracy'], 'c-o', label="Validation Accuracy")

        plt.title("Training & Validation Loss and Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        plt.xticks(range(1, epochs + 1))
        plt.show()

        output_dir = './model_save/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        api = HfApi()
        repo_url = api.create_repo(repo_id=job_id, token=hf_token)
        repo = Repository(local_dir=f"models/{job_id}", clone_from=repo_url, token=hf_token)

        model_to_save.save_pretrained(repo.local_dir)
        tokenizer.save_pretrained(repo.local_dir)

        repo.git_add(pattern=".")
        repo.git_commit("Add fine-tuned model files")
        repo.git_push()

        # Capture the end time
        pipeline_end_time = time.time()
        total_pipeline_time = format_time(pipeline_end_time - pipeline_start_time)

        return repo_url, avg_val_loss, avg_val_accuracy, total_pipeline_time

    except Exception as e:
        await update_job_status(job_id, 'pending')
        raise RuntimeError(f"Training pipeline encountered an error: {str(e)}")

    finally:
        # Clean up the dataset directory
        shutil.rmtree(dataset_dir)
