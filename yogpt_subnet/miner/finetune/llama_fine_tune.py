import asyncio
import os
import shutil
import torch
from datasets import load_dataset
from yogpt_subnet.miner.utils.helpers import update_job_status #type:ignore
from huggingface_hub import HfApi, Repository
from peft import LoraConfig
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments)
from trl import SFTTrainer

new_model_name = "llama2_model_finetuned"

async def fine_tune_llama(base_model, dataset_id, new_model_name, hf_token, job_id):
    """Train a model with the given parameters and upload it to Hugging Face."""
    base_model = str(base_model)
    print("------basemodel specified-----" + base_model)
    print(".......new_model_name ........" + new_model_name)
    print(".......dataset specified ........" + dataset_id)
    # Designate directories
    dataset_dir = os.path.join("data", dataset_id)
    os.makedirs(dataset_dir, exist_ok=True)
import asyncio
import os
import shutil
import torch
from datasets import load_dataset
from yogpt_subnet.miner.utils.helpers import update_job_status #type:ignore
from huggingface_hub import HfApi, Repository
from peft import LoraConfig
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments)
from trl import SFTTrainer

new_model_name = "llama2_model_finetuned"

async def fine_tune_llama(base_model, dataset_id, new_model_name, hf_token, job_id):
    """Train a model with the given parameters and upload it to Hugging Face."""
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

        # Set compute dtype
        compute_dtype = getattr(torch, "float16")

        # Load quantization config
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=False,
        )

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quant_config,
            device_map={"": 0}
        )
        model.config.use_cache = False
        model.config.pretraining_tp = 1

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # Load LoRA configuration
        peft_args = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Set training parameters
        training_params = TrainingArguments(
            output_dir="training_output",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            optim="paged_adamw_32bit",
            save_steps=25,
            logging_steps=25,
            learning_rate=2e-5,
            weight_decay=0.001,
            fp16=True,
            bf16=False,

            max_grad_norm=0.3,
            max_steps=-1,
            warmup_ratio=0.03,
            group_by_length=True,
            lr_scheduler_type="constant",
            report_to="tensorboard",
            save_total_limit=1,  # Save only the last checkpoint
        )

        # Set supervised fine-tuning parameters
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=peft_args,
            dataset_text_field="text",
            max_seq_length=512,
            tokenizer=tokenizer,
            args=training_params,
            packing=False,
        )

        # Train model
        try:
            trainer.train()
        except Exception as e:
            await update_job_status(job_id, 'pending')  # Update status back to pending
            raise RuntimeError(f"Training failed: {str(e)}")

        # Create repository on Hugging Face and clone it locally
        api = HfApi()
        repo_url = api.create_repo(repo_id=new_model_name, token=hf_token)
        repo = Repository(local_dir=f"models/{new_model_name}", clone_from=repo_url, token=hf_token)

        # Save trained model locally in the cloned directory
        trainer.model.save_pretrained(repo.local_dir)
        trainer.tokenizer.save_pretrained(repo.local_dir)

        # Add all files to the git repository, commit, and push
        repo.git_add(pattern=".")
        repo.git_commit("Add fine-tuned model files")
        repo.git_push()

        return repo_url

    except Exception as e:
        await update_job_status(job_id, 'pending')
        raise RuntimeError(f"Training pipeline encountered an error: {str(e)}")

    finally:
        # Clean up the dataset directory
        shutil.rmtree(dataset_dir)
    try:
        # Load dataset
        dataset = load_dataset(dataset_id, split="train", cache_dir=dataset_dir)

        # Set compute dtype
        compute_dtype = getattr(torch, "float16")

        # Load quantization config
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=False,
        )

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quant_config,
            device_map={"": 0}
        )
        model.config.use_cache = False
        model.config.pretraining_tp = 1

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # Load LoRA configuration
        peft_args = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Set training parameters
        training_params = TrainingArguments(
            output_dir="training_output",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            optim="paged_adamw_32bit",
            save_steps=25,
            logging_steps=25,
            learning_rate=2e-5,
            weight_decay=0.001,
            fp16=True,
            bf16=False,
            max_grad_norm=0.3,
            max_steps=-1,
            warmup_ratio=0.03,
            group_by_length=True,
            lr_scheduler_type="constant",
            report_to="tensorboard",
            save_total_limit=1,  # Save only the last checkpoint
        )

        # Set supervised fine-tuning parameters
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=peft_args,
            dataset_text_field="text",
            max_seq_length=512,
            tokenizer=tokenizer,
            args=training_params,
            packing=False,
        )

        # Train model
        try:
            trainer.train()
        except Exception as e:
            await update_job_status(job_id, 'pending')  # Update status back to pending
            raise RuntimeError(f"Training failed: {str(e)}")

        # Create repository on Hugging Face and clone it locally
        api = HfApi()
        repo_url = api.create_repo(repo_id=new_model_name, token=hf_token)
        repo = Repository(local_dir=f"models/{new_model_name}", clone_from=repo_url, token=hf_token)

        # Save trained model locally in the cloned directory
        trainer.model.save_pretrained(repo.local_dir)
        trainer.tokenizer.save_pretrained(repo.local_dir)

        # Add all files to the git repository, commit, and push
        repo.git_add(pattern=".")
        repo.git_commit("Add fine-tuned model files")
        repo.git_push()

        return repo_url

    except Exception as e:
        await update_job_status(job_id, 'pending')
        raise RuntimeError(f"Training pipeline encountered an error: {str(e)}")

    finally:
        # Clean up the dataset directory
        shutil.rmtree(dataset_dir)