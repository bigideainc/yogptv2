import datetime
import os
import shutil
import time
import bitsandbytes as bnb
import evaluate
import numpy as np
import torch
import json
import wandb
import asyncio
from typing import Tuple
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig, DataCollatorForSeq2Seq,TrainingArguments)
from trl import SFTConfig, SFTTrainer
from huggingface_hub import HfApi, login
import uuid
from utils.HFManager import commit_to_central_repo
from utils.wandb_initializer import initialize_wandb
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


async def fine_tune_llama(dataset_id,epochs, batch_size, learning_rate,hf_token, job_id, miner_uid):
    """Train a model with the given parameters and upload it to Hugging Face."""
    base_model = str("NousResearch/Llama-2-7b-chat-hf")

    try:
        # Login to Hugging Face
        login(hf_token)
        hf_api = HfApi(hf_token)
        # Initialize wandb
        wandb_run = initialize_wandb(job_id, miner_uid)
        if wandb_run:
            print("Weights & Biases run initialized successfully.")
        else:
            print("Weights & Biases run initialization failed.")
        # Load dataset
        

        # Set model configuration for Kaggle compatibility
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            base_model, token=hf_token, trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # Initialize the model
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            token=hf_token,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map='auto'
        )

        # Enable gradient checkpointing
        model.config.use_cache = False
        model.config.pretraining_tp = 1
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

        def tokenize_function(examples):
            inputs = examples['text']
            model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=512)
            # Set up labels for language modeling
            model_inputs["labels"] = model_inputs["input_ids"].copy()  
            return model_inputs
        
        dataset = load_dataset(dataset_id, split="train",token=hf_token)
        # Split dataset into training and validation sets (90% train, 10% validation)
        dataset = dataset.map(tokenize_function, batched=True)
        split_dataset = dataset.train_test_split(test_size=0.1)
        train_dataset, eval_dataset = split_dataset["train"], split_dataset["test"]

        # PEFT Model Setup
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=64,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none"
        )
        model = get_peft_model(model, peft_config)

        # Training arguments
        training_args = TrainingArguments(
            output_dir="./results",  
            per_device_train_batch_size=batch_size,   
            gradient_accumulation_steps=16,  
            learning_rate=learning_rate,             
            num_train_epochs=epochs,                
            optim="paged_adamw_8bit",       
            fp16=True, 
            logging_steps=10, 
            eval_strategy="steps",
            save_strategy="steps",
            eval_steps=50,
            save_steps=500,
            save_total_limit=3,
            load_best_model_at_end=True,                    
            run_name=f"training-{job_id}-{miner_uid}" ,    
            report_to="wandb"
        )

        # Data collator
        data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True, 
            label_pad_token_id=tokenizer.pad_token_id)

        # Trainer setup
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        # Train model
        start_time = time.time()
        train_result = trainer.train()
        final_loss = train_result.training_loss
        total_training_time = time.time() - start_time
        if wandb_run:
            wandb_run.log({
                "final_loss": final_loss ,
                "training_time": total_training_time,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
            })
        else:
            print("Weights & Biases run was not initialized. Skipping wandb logging.")

        # Push Model to Hugging Face
        repo_name = f"finetuned-{base_model}-{job_id}-{int(time.time())}"
        repo_url = hf_api.create_repo(repo_name, private=False)
        model.push_to_hub(repo_name, token=hf_token)
        tokenizer.push_to_hub(repo_name, token=hf_token)

        metrics.update({
            "training_time": total_training_time,
            "model_repo": repo_url,
            "final_loss": final_loss ,
            "miner_uid": miner_uid,
            "datasetid":dataset_id,
            "total_epochs": epochs,
            "job_id": job_id,
        })

        commit_to_central_repo(repo_url, metrics, miner_uid)
        wandb_run.finish()
        print(f"Model training and upload is completed")
        print(f"Your trained model has been uplodaed to {repo_url} repository")
        print(f"Your training results have been submitted to the validator")
        return metrics

    except Exception as e:
        if wandb.run:
            wandb.alert(title="Pipeline Error", text=str(e))
            wandb.finish()
        raise RuntimeError(f"Pipeline encountered an error: {str(e)}")

    finally:
        try:
            shutil.rmtree("./results")
        except Exception as cleanup_error:
            print(f"Cleanup error: {cleanup_error}")
