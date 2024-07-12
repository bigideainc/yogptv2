import os
import shutil
import sys
import time

# Append directories to sys.path for relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../', 'dataset')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../', 'model')))

from datasets import load_dataset
from yogpt_subnet.miner.utils.helpers import update_job_status #type:ignore
from huggingface_hub import HfApi, Repository
from transformers import (GPT2LMHeadModel, GPT2Tokenizer, Trainer,TrainingArguments,DataCollatorForLanguageModeling)

new_model_name="gpt2_model_finetuned"

async def fine_tune_gpt(job_id, base_model, dataset_id, new_model_name, hf_token):
    """Fine-tunes a GPT-2 model and uploads it to Hugging Face Hub."""
    # print(f"Transformer version: {transformers.__version__}")
    start_time = time.time()
    final_loss = None
    # Designate directories
    base_model = str(base_model)
    print("------basemodel specified-----" + base_model)
    print(".......new_model_name ........" + new_model_name)
    print(".......dataset specified ........" + dataset_id)
    
    dataset_dir = os.path.join("data", dataset_id)
    os.makedirs(dataset_dir, exist_ok=True)

    try:
        # Load dataset
        dataset = load_dataset(dataset_id, split="train", cache_dir=dataset_dir,trust_remote_code=True)

        # Load GPT-2 tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained(base_model, pad_token="<|endoftext|>")

        # Load GPT-2 model
        model = GPT2LMHeadModel.from_pretrained(base_model)

        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

        print("...... dataset loaded................")
        # Set training parameters
        def compute_loss(model, inputs):
            labels = inputs.pop("labels")  # Remove labels from inputs
            outputs = model(**inputs, labels=labels)  # Forward pass with labels
            return outputs.loss 
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=1,
            per_device_train_batch_size=4,
            warmup_steps=500,
            weight_decay=0.01,
            label_names=['input_ids', 'attention_mask'], 
            logging_dir='./logs',
            eval_steps=500,
            logging_steps=500,
            fp16=False,
            report_to="tensorboard",
        )

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_loss,
            data_collator=data_collator
        )

        # Train model
        print(".......Training starting .......... ")
        trainer.train()
        final_loss = trainer.state.log_history[-1].get('loss')
        print(".......Training ended .......... ")
        # Create repository on Hugging Face and clone it locally
        api = HfApi()
        repo_url = api.create_repo(repo_id=new_model_name, token=hf_token)
        repo = Repository(local_dir=f"models/{new_model_name}", clone_from=repo_url, token=hf_token)

        # Save trained model locally in the cloned directory
        trainer.save_model(repo.local_dir)

        # Add all files to the git repository, commit, and push
        repo.git_add(pattern=".")
        repo.git_commit("Add fine-tuned model files")
        repo.git_push()
        total_time = time.time() - start_time
        return repo_url,total_time,final_loss

    except Exception as e:
        await update_job_status(job_id, 'pending')
        raise RuntimeError(f"Training pipeline encountered an error: {str(e)}")

    finally:
        # Clean up the dataset directory
        shutil.rmtree(dataset_dir)

# Example usage
# import asyncio

# job_id = "job1234"
# base_model = "gpt2"
# dataset_id = "wikitext-2-raw-v1"  # Example dataset configuration
# new_model_name = "toby_gpt2"
# hf_token = "your_hugging_face_token_here"

# asyncio.run(fine_tune_gpt(job_id, base_model, dataset_id, new_model_name, hf_token))