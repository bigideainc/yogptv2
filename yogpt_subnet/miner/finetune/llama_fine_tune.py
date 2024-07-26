import os
import shutil
import time
import datetime

import bitsandbytes as bnb
import evaluate
import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import HfApi, Repository, create_repo, login, whoami
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,DataCollatorForSeq2Seq,TrainingArguments,AutoModelForCausalLM,BitsAndBytesConfig)
from trl import SFTTrainer,SFTConfig
from yogpt_subnet.miner.utils.helpers import update_job_status


def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))
async def fine_tune_llama(base_model, dataset_id, new_model_name, hf_token, job_id):
    """Train a model with the given parameters and upload it to Hugging Face."""
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

        # Load dataset
        dataset = load_dataset(dataset_id, split="train", use_auth_token=hf_token)

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
        tokenizer.padding_side = "right"

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

        # Training arguments
        training_args = TrainingArguments(
            output_dir="./results",  
            per_device_train_batch_size=4,  
            per_device_eval_batch_size=4,   
            gradient_accumulation_steps=2,  
            learning_rate=2e-5,             
            max_steps=100,                 
            optim="paged_adamw_8bit",       
            fp16=True,                      
            run_name="llama-2-guanaco",     
            report_to="none"
        )

        # Data collator
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

        # Define compute metrics
        accuracy_metric = evaluate.load("accuracy")
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
            loss = np.mean(logits - labels)  # Dummy calculation for loss
            return {"accuracy": accuracy["accuracy"], "loss": loss}

        # Trainer setup
        trainer = SFTTrainer(
            max_seq_length=None,
            model=peft_model,
            dataset_text_field="text",
            args=training_args,
            peft_config=peft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
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
        accuracy = 0
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

        # Capture the end time
        pipeline_end_time = time.time()
        total_pipeline_time = format_time(pipeline_end_time - pipeline_start_time)

        return repo_url, loss, accuracy, total_pipeline_time

    except Exception as e:
        await update_job_status(job_id, 'pending')
        raise RuntimeError(f"Training pipeline encountered an error: {str(e)}")

    finally:
        # Clean up the dataset directory
        shutil.rmtree(dataset_dir)
