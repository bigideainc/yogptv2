import datetime
import os
import shutil
import time
import torch
import wandb
from trl import SFTTrainer
from datasets import load_dataset
from huggingface_hub import login
from transformers import (AutoTokenizer, AutoModelForCausalLM,TrainingArguments,BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model

from yogpt_subnet.miner.models.storage.hugging_face_store import \
    HuggingFaceModelStore
from yogpt_subnet.miner.utils.helpers import update_job_status


def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

async def fine_tune_gemma(base_model, dataset_id, new_model_name, hf_token, job_id):
    """Fine-tune gemma model and upload it to Hugging Face."""
    print("Starting fine-tuning process...")
    base_model = str(base_model)
    print("*------base model specified-----*" + base_model)
    print("*.......new model name ........*" + new_model_name)
    print("*.......dataset specified ........*" + dataset_id)

    pipeline_start_time = time.time()
    dataset_dir = os.path.join("data", dataset_id)
    os.makedirs(dataset_dir, exist_ok=True)
    lora_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    

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

        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=lora_config)
        peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=4,
        lora_alpha=16,
        lora_dropout=0.01,
        )
        peft_model = get_peft_model(model, peft_config)

        tokenized_dataset = dataset.map(lambda samples: tokenizer(samples["quote"]), batched=True)
        print("dataset tokenized")
        def formatting_func(example):
            text = f"Quote: {tokenized_dataset['train']['quote'][0]}\nAuthor: {tokenized_dataset['train']['author'][0]}"
            return [text]
        print("..... training starting ......")
        training_args = TrainingArguments(
            output_dir="./fine-tuned_model",
            overwrite_output_dir=True,
            num_train_epochs=1000,
            per_device_train_batch_size=4, 
            gradient_accumulation_steps=4, 
            learning_rate=2e-4,
            fp16=True,  
            logging_steps=100,
            optim="paged_adamw_8bit"
        )
        print("Setting up trainer........")
        trainer = SFTTrainer(
            model=peft_model,
            train_dataset=tokenized_dataset,
            args=training_args,
            peft_config=peft_config,
            formatting_func=formatting_func,
        )
        print("Trainer set up successfully.")

        print("Starting training...")

        # Train model
        try:
            train_result = trainer.train()
            loss = train_result.training_loss
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
        print(loss)
        print(accuracy)
        print(total_pipeline_time)
        return repo_url, loss, accuracy, total_pipeline_time

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
