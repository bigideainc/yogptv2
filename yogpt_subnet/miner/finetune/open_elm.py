import asyncio
import os
import shutil
import sys
import time  # Import the time module
import uuid
import datetime

import torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          TrainingArguments, set_seed)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer, setup_chat_format

from yogpt_subnet.miner.models.storage.hugging_face_store import \
    HuggingFaceModelStore
from yogpt_subnet.miner.utils.helpers import update_job_status

# Append directories to sys.path for relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../', 'dataset')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../', 'model')))

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

async def fine_tune_openELM(base_model,dataset_id,new_model_name,hf_token,job_id):
    try:
        # Capture the start time
        pipeline_start_time = time.time()
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            token=hf_token,
            use_cache=False
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "TinyPixel/Llama-2-7B-bf16-sharded",
            trust_remote_code=True,
            use_fast=False,
            token=hf_token
        )

        set_seed(42)
        lr = 5e-5
        run_id = f"OpenELM-1_IB_LR-{lr}_OA_{str(uuid.uuid4())}"                                                                                                                                                                                                                                                                                                                                         

        # Setup chat format
        model, tokenizer = setup_chat_format(model, tokenizer)
        if tokenizer.pad_token in [None, tokenizer.eos_token]:
            tokenizer.pad_token = tokenizer.unk_token

        # Load dataset                                                                                                                                                                                                                                                                                                                                                                  
        dataset = load_dataset(dataset_id, use_auth_token=hf_token)

        # Training arguments
        training_arguments = TrainingArguments(
            output_dir=f"out_{run_id}",
            evaluation_strategy="steps",
            label_names=["labels"],
            per_device_train_batch_size=16,
            gradient_accumulation_steps=8,
            save_steps=250,
            eval_steps=250,
            logging_steps=1,
            learning_rate=lr,
            num_train_epochs=3,
            lr_scheduler_type="constant",
            optim='paged_adamw_8bit',
            bf16=False,
            report_to="tensorboard",
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            group_by_length=True,
        )

        # Trainer initialization
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset['test'],
            data_collator=DataCollatorForCompletionOnlyLM(
                instruction_template="user",
                response_template="assistant",
                tokenizer=tokenizer,
                mlm=False
            ),
            max_seq_length=2048,
            dataset_kwargs=dict(add_special_tokens=False),
            args=training_arguments,
        )

        # Train model
        train_result = trainer.train()

        # Evaluate model
        eval_result = trainer.evaluate()

        # Collect metrics
        train_loss = train_result.training_loss
        eval_loss = eval_result['eval_loss']
        accuracy = 0

        # Create repository on Hugging Face and clone it locally
        store = HuggingFaceModelStore()
        repo_url = store.upload_model(model, tokenizer, job_id)

        # Capture the end time
        pipeline_end_time = time.time()
        total_pipeline_time = format_time(pipeline_end_time - pipeline_start_time)
        print("........ model details...........")
        print(repo_url)
        print(eval_loss)
        print(accuracy)
        print(total_pipeline_time)
        return repo_url, eval_loss, accuracy, total_pipeline_time

    except Exception as e:
        # Handle exceptions and update job status
        await update_job_status(job_id, 'pending')
        return None, None, None, None
    

    finally:
        # Clean up any resources if needed
        pass  # No specific cleanup needed in this example
