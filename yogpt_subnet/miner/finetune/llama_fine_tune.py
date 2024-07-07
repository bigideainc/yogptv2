import asyncio
import os
import shutil

import bitsandbytes as bnb
import evaluate
import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import HfApi, Repository, create_repo, login, whoami
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, Trainer, TrainingArguments)

from yogpt_subnet.miner.utils.helpers import update_job_status


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
        # Login to Hugging Face
        login(hf_token)

        # Load dataset
        dataset = load_dataset(dataset_id, split="train")

        # Split dataset into training and validation sets (90% train, 10% validation)
        split_dataset = dataset.train_test_split(test_size=0.1)
        train_dataset = split_dataset['train']
        eval_dataset = split_dataset['test']

        # Set model configuration for Kaggle compatibility
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_has_fp16_weight=False,
        )

        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            base_model, use_auth_token=hf_token, trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token

        # Initialize the model
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            use_auth_token=hf_token,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map='auto'
        )
        model.config.use_cache = False
        model.config.pretraining_tp = 1

        # Enable gradient checkpointing
        model.supports_gradient_checkpointing = True  
        model.gradient_checkpointing_enable()     
        model.enable_input_require_grads()    
        model.config.use_cache = False

        # Find all linear layers
        def find_all_linear_names(model):
            cls = bnb.nn.Linear4bit
            lora_module_names = set()
            for name, module in model.named_modules():
                if isinstance(module, cls):
                    names = name.split('.')
                    lora_module_names.add(names[0] if len(names) == 1 else names[-1])
            if 'lm_head' in lora_module_names:
                lora_module_names.remove('lm_head')
            return list(lora_module_names)

        lora_modules = find_all_linear_names(model)

        # PEFT Model Setup
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=lora_modules
        )
        peft_model = get_peft_model(model, peft_config)
        peft_model.is_parallelizable = True
        peft_model.model_parallel = True
        peft_model.print_trainable_parameters()

        # Set training parameters
        training_params = TrainingArguments(
            output_dir="training_output",
            num_train_epochs=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            optim="paged_adamw_32bit",
            save_steps=25,
            logging_steps=25,
            learning_rate=2e-5,
            weight_decay=0.001,
            fp16=False,
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
            max_seq_length=None,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        # Train model
        try:
            train_result = trainer.train()
            metrics = train_result.metrics
            trainer.save_metrics("train", metrics)
            trainer.save_state()
        except Exception as e:
            await update_job_status(job_id, 'pending')  # Update status back to pending
            raise RuntimeError(f"Training failed: {str(e)}")

        # Evaluate the model on the validation set
        eval_metrics = trainer.evaluate()
        accuracy = eval_metrics.get("eval_accuracy")
        loss = eval_metrics.get("eval_loss")

        # Create repository on Hugging Face and clone it locally
        api = HfApi()
        repo_url = api.create_repo(repo_id=job_id, token=hf_token)
        repo = Repository(local_dir=f"models/{job_id}", clone_from=repo_url, token=hf_token)

        # Save trained model locally in the cloned directory
        trainer.model.save_pretrained(repo.local_dir)
        trainer.tokenizer.save_pretrained(repo.local_dir)

        # Add all files to the git repository, commit, and push
        repo.git_add(pattern=".")
        repo.git_commit("Add fine-tuned model files")
        repo.git_push()

        return repo_url, loss, accuracy

    except Exception as e:
        await update_job_status(job_id, 'pending')
        raise RuntimeError(f"Training pipeline encountered an error: {str(e)}")

    finally:
        # Clean up the dataset directory
        shutil.rmtree(dataset_dir)
