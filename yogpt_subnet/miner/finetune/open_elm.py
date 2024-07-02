import asyncio
import os
import shutil
import sys
import uuid

import torch
from datasets import load_dataset
from helpers import update_job_status
from storage.hugging_face_store import HuggingFaceModelStore
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          TrainingArguments, get_constant_schedule, set_seed)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer, setup_chat_format

# Append directories to sys.path for relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../', 'dataset')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../', 'model')))


async def fine_tune_openELM(job_id, base_model, dataset_id, new_model_name, hf_token):
    try:
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            trust_remote_code=True,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            torch_dtype=torch.float16,
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True,
        )

        set_seed(42)
        lr = 5e-5
        run_id = f"OpenELM-1_IB_LR-{lr}_OA_{str(uuid.uuid4())}"

        # Setup chat format
        model, tokenizer = setup_chat_format(model, tokenizer)
        if tokenizer.pad_token in [None, tokenizer.eos_token]:
            tokenizer.pad_token = tokenizer.unk_token

        # Load dataset
        dataset = load_dataset(dataset_id)

        # Training arguments
        training_arguments = TrainingArguments(
            output_dir=f"out_{run_id}",
            evaluation_strategy="steps",
            label_names=["labels"],
            per_device_train_batch_size=8,
            gradient_accumulation_steps=2,
            save_steps=250,
            eval_steps=250,
            logging_steps=1,
            learning_rate=lr,
            num_train_epochs=3,
            lr_scheduler_type="constant",
            optim='paged_adamw_8bit',
            bf16=False,
            gradient_checkpointing=True,
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
        trainer.train()

        # Create repository on Hugging Face and clone it locally
        store = HuggingFaceModelStore()
        await store.upload_model(model, tokenizer, job_id)

    except Exception as e:
        # Handle exceptions and update job status
        await update_job_status(job_id, 'failed')
        raise RuntimeError(f"Training pipeline encountered an error: {str(e)}")

    finally:
        # Clean up any resources if needed
        pass  # No specific cleanup needed in this example
