import asyncio
import os
import signal
import sys
import json
from typing import Optional

import uvicorn
from dotenv import load_dotenv
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.text import Text
import pyfiglet

from communex.module.module import Module
from yogpt_subnet.miner.auth.auth import authenticate
from yogpt_subnet.miner.finetune.runpod.pipeline import generate_pipeline_script
from yogpt_subnet.miner.utils.helpers import (
    fetch_jobs, fetch_and_save_job_details, update_job_status,
    register_completed_job, submit_to_runpod
)
from yogpt_subnet.miner.finetune.llama_fine_tune import fine_tune_llama
from yogpt_subnet.miner.finetune.gpt_fine_tune import fine_tune_gpt
from yogpt_subnet.miner.finetune.open_elm import fine_tune_openELM
miner_id
class Config:
    def __init__(self):
        load_dotenv()
        self.base_url = os.getenv("BASE_URL")
        self.hf_access_token = os.getenv("HF_ACCESS_TOKEN")
        self.token = os.getenv("TOKEN")
        self.miner_id = os.getenv("MINER_ID")
        self.run_on_runpod: bool = os.getenv('RUN_ON_RUNPOD', 'false').lower() == 'true'
        self.runpod_api_key: Optional[str] = os.getenv('RUNPOD_API_KEY')

        if not self.base_url or not self.hf_access_token:
            raise ValueError("BASE_URL and HF_ACCESS_TOKEN must be set in the environment or .env file")

        if self.run_on_runpod and not self.runpod_api_key:
            raise ValueError("RUNPOD_API_KEY must be set if RUN_ON_RUNPOD is true")

class Trainer(Module):
    def __init__(self):
        super().__init__()
        self.console = Console()
        self.config = Config()
        self.current_job_id: Optional[str] = None

    def display_welcome_message(self):
        fig = pyfiglet.Figlet(font='slant')
        welcome_text = fig.renderText('YOGPT Miner')
        self.console.print(welcome_text, style="bold blue")
        self.console.print(Panel(Text("Welcome to YOGPT Miner System!", justify="center"), style="bold green"))

    def get_user_credentials(self):
        username = Prompt.ask("Enter your username")
        password = Prompt.ask("Enter your password", password=True)
        return username, password

    def authenticate_user(self, username: str, password: str) -> bool:
        token, miner_id = authenticate(username, password)
        if not token or not miner_id:
            self.console.print("[bold red]Authentication failed.[/bold red]")
            return False
        self.config.token, self.config.miner_id = token, miner_id
        return True

    async def process_job(self, job_details: dict):
        model_id = job_details['baseModel']
        dataset_id = job_details['huggingFaceId']
        job_id = job_details['jobId']
        new_model_name = job_id
        self.current_job_id = job_id

        try:
            if self.config.run_on_runpod:
                script_path = f"pipeline_script_{job_id}.py"
                generate_pipeline_script(job_details, script_path)
                submit_to_runpod(script_path, self.config.runpod_api_key)
            else:
                model_repo_url, loss, accuracy, total_pipeline_time = None, None, None, None
                model_detected = model_id.lower()

                if 'llama' in model_detected:
                    model_repo_url, loss, accuracy, total_pipeline_time = await fine_tune_llama(
                        model_id, dataset_id, new_model_name, self.config.hf_access_token, job_id
                    )
                elif 'gpt' in model_detected:
                    model_repo_url, loss, accuracy, total_pipeline_time = await fine_tune_gpt(
                        model_id, dataset_id, new_model_name, self.config.hf_access_token, job_id
                    )
                elif 'openelm' in model_detected:
                    model_repo_url, loss, accuracy, total_pipeline_time = await fine_tune_openELM(
                        model_id, dataset_id, new_model_name, self.config.hf_access_token, job_id
                    )
                else:
                    raise ValueError(f"Unsupported model ID: {model_id}")

                if model_repo_url:
                    self.console.log(f"Model uploaded to: {model_repo_url}")
                    self.console.log(f"Total pipeline time: {total_pipeline_time}")
                    await update_job_status(job_id, 'completed', self.config.base_url, self.config.token)
                    await register_completed_job(job_id, model_repo_url, loss, accuracy, total_pipeline_time,
                                                 self.config.base_url, self.config.token)
                else:
                    raise RuntimeError(f"Failed to process job {job_id}: No model URL returned")

        except Exception as e:
            self.console.log(f"Error processing job {job_id}: {str(e)}")
            await update_job_status(job_id, 'failed', self.config.base_url, self.config.token)
        finally:
            self.current_job_id = None

    async def main_loop(self):
        while True:
            try:
                jobs = await fetch_jobs(self.config.base_url, self.config.token)
                if jobs:
                    job_id = jobs[0]['id']
                    self.console.log(f"Executing JobId: {job_id}")
                    job_details_path = await fetch_and_save_job_details(job_id, self.config.base_url, 
                                                                        self.config.token, self.config.miner_id)
                    if job_details_path:
                        with open(job_details_path, 'r') as file:
                            job_details = json.load(file)
                        await self.process_job(job_details)
                await asyncio.sleep(2)
            except Exception as e:
                self.console.log(f"Error in main loop: {str(e)}")
                await asyncio.sleep(5)

    def handle_interrupt(self, signal, frame):
        if self.current_job_id:
            asyncio.run(update_job_status(self.current_job_id, 'pending', 
                                          self.config.base_url, self.config.token))
        self.console.log("[bold red]Interrupted. Exiting...[/bold red]")
        sys.exit(0)

    async def main(self):
        self.display_welcome_message()

        while True:
            username, password = self.get_user_credentials()
            if self.authenticate_user(username, password):
                self.console.print("[bold green]Authentication successful. Starting server...[/bold green]")
                break
            retry = Prompt.ask("Authentication failed. Do you want to try again?", choices=["yes", "no"], default="yes")
            if retry.lower() != "yes":
                self.console.print("Exiting program.")
                return

        signal.signal(signal.SIGINT, self.handle_interrupt)
        
        # Start your Uvicorn server here if needed
        # uvicorn.run(app, host="0.0.0.0", port=8889)

        await self.main_loop()

if __name__ == "__main__":
    trainer = Trainer()
    asyncio.run(trainer.run())