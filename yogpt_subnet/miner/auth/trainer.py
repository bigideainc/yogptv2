import argparse
import asyncio
import json
import os
import signal
import ssl
import sys
import time

import aiohttp
import pyfiglet
from communex.module.module import Module, endpoint
from dotenv import load_dotenv
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from yogpt_subnet.miner.finetune.runpod.pipeline import \
    generate_pipeline_script

# Append directories to sys.path for relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../', 'finetune')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './', 'auth')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'finetune/runpod')))
from yogpt_subnet.miner.auth.auth import authenticate  # type:ignore
from yogpt_subnet.miner.finetune.gpt_fine_tune import \
    fine_tune_gpt  # type:ignore
from yogpt_subnet.miner.finetune.llama_fine_tune import \
    fine_tune_llama  # type:ignore
from yogpt_subnet.miner.finetune.open_elm import \
    fine_tune_openELM  # type:ignore
from yogpt_subnet.miner.utils.helpers import (  # type:ignore
    fetch_and_save_job_details, fetch_jobs, register_completed_job,
    submit_to_runpod, update_job_status)


class Trainer(Module):
    def __init__(self):
        load_dotenv()
        self.BASE_URL = os.getenv("BASE_URL")
        self.HF_ACCESS_TOKEN = os.getenv("HF_ACCESS_TOKEN")
        self.TOKEN = os.getenv("TOKEN")
        self.MINER_ID = os.getenv("MINER_ID")
        self.console = Console()
        self.current_job_id = None  # Global variable to store the current job ID

    @endpoint
    def display_welcome_message(self):
        fig = pyfiglet.Figlet(font='slant')
        welcome_text = fig.renderText('YOGPT Miner')
        self.console.print(welcome_text, style="bold blue")
        self.console.print(Panel(Text("Welcome to YOGPT Miner System!", justify="center"), style="bold green"))

    @endpoint
    async def fetch_jobs(self):
        timeout = aiohttp.ClientTimeout(total=120)  # Increased timeout to 120 seconds
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
            headers = {'Authorization': f'Bearer {self.TOKEN}'}
            try:
                async with session.get(f"{self.BASE_URL}/pending-jobs", headers=headers) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 401:
                        return "Unauthorized"
                    else:
                        self.console.log(f"Failed to fetch jobs: {await response.text()}")
                        return []
            except aiohttp.ClientConnectorError as e:
                self.console.log(f"Connection error: {e}")
                return []
            
    async def fetch_and_save_job_details(self, job_id):
        async with aiohttp.ClientSession() as session:
            headers = {'Authorization': f'Bearer {self.TOKEN}'}
            async with session.post(f"{self.BASE_URL}/start-training/{job_id}", headers=headers, json={'minerId': self.MINER_ID}) as response:
                if response.status == 200:
                    job_details = await response.json()
                    job_dir = os.path.join(os.getcwd(), 'jobs', job_id)
                    os.makedirs(job_dir, exist_ok=True)
                    details_path = os.path.join(job_dir, 'details.json')
                    job_details['jobId'] = job_id
                    with open(details_path, 'w') as f:
                        json.dump(job_details, f, indent=2)
                    return details_path
                else:
                    self.console.log(f"Failed to start training for job {job_id}: {await response.text()}")
                    return None
 
    async def update_job_status(self, job_id, status):
        url = f"{self.BASE_URL}/update-status/{job_id}"
        async with aiohttp.ClientSession() as session:
            headers = {'Authorization': f'Bearer {self.TOKEN}', 'Content-Type': 'application/json'}
            async with session.patch(url, json={'status': status}, headers=headers) as response:
                try:
                    if response.status == 200:
                        self.console.log(f"Status updated to {status} for job {job_id}")
                    else:
                        response.raise_for_status()
                except aiohttp.ClientResponseError as err:
                    self.console.log(f"Failed to update status for job {job_id}: {err}")
                except Exception as e:
                    self.console.log(f"An error occurred: {e}")
  
    async def process_job(self, job_details, run_on_runpod=False, runpod_api_key=None):
        global current_job_id
        model_id = job_details['baseModel']
        dataset_id = job_details['huggingFaceId']
        job_id = job_details['jobId']
        new_model_name = job_id
        self.current_job_id = job_id  # Update the current job ID

        try:
            if run_on_runpod:
                if runpod_api_key is None:
                    raise ValueError("RunPod API key is required when running on RunPod.")
                script_path = f"pipeline_script_{job_id}.py"
                generate_pipeline_script(job_details, script_path)
                submit_to_runpod(script_path, runpod_api_key)
            else:
                model_repo_url, loss, accuracy = None, None, None, None
                self.console.log(f"General model {model_id}")
                model_detected = model_id.lower()
                if 'llama' in model_detected:
                    self.console.log(f"model is of type Llama: {model_id}")
                    model_repo_url, loss, accuracy = await fine_tune_llama(model_id, dataset_id, new_model_name, self.HF_ACCESS_TOKEN, job_id)
                elif 'gpt' in model_detected:
                    self.console.log(f"model is  of type GPT:"+model_id)
                    model_repo_url = await fine_tune_gpt(model_id, dataset_id, new_model_name, self.HF_ACCESS_TOKEN, job_id)
                elif 'openelm' in model_detected:
                    self.console.log(f"model is  of type OpenELM:"+model_id)
                    model_repo_url = await fine_tune_openELM(model_id, dataset_id, new_model_name, self.HF_ACCESS_TOKEN, job_id)
                else:
                    self.console.log(f"Unsupported model ID: {model_id}. Skipping job.")
                    return  # Skip this job and proceed to the next one

                self.console.log(f"Model uploaded to: {model_repo_url}")

            await self.update_job_status(job_id, 'completed')
            await register_completed_job(job_id, model_repo_url, loss, accuracy)  # Pass the correct metrics
        except RuntimeError as e:
            self.console.log(f"Failed to process job {job_id}: {str(e)}")
        except Exception as e:
            await self.update_job_status(job_id, 'pending')
            self.console.log(f"Unexpected error occurred while processing job {job_id}: {str(e)}")
        finally:
            self.current_job_id = None  # Reset the current job ID
 
    def handle_interrupt(self, signal, frame):
        if self.current_job_id:
            asyncio.run(self.update_job_status(self.current_job_id, 'pending'))
        self.console.log("[bold red]Interrupted. Exiting...[/bold red]")
        sys.exit(0)

    async def main(self):
        self.display_welcome_message()

        username = input("Enter your username: ")
        password = input("Enter your password: ")

        token, miner_id = authenticate(username, password)
        if not token or not miner_id:
            self.console.log("[bold red]Authentication failed. Exiting...[/bold red]")
            sys.exit(1)

        self.TOKEN, self.MINER_ID = token, miner_id

        progress_table = Table.grid(expand=True)
        progress_table.add_column(justify="center", ratio=1)
        status_panel = Panel("Waiting for training jobs...", title="Status", border_style="green")
        progress_table.add_row(status_panel)

        with Live(progress_table, refresh_per_second=10, console=self.console) as live:
            while True:
                live.update(progress_table)
                jobs = await self.fetch_jobs()

                if jobs == "Unauthorized":
                    self.console.log("[bold red]Unauthorized access. Please check your credentials.[/bold red]")
                    break
                elif jobs:
                    job_id = jobs[0]['id']
                    self.console.log(f"Executing JobId: {job_id}")
                    job_details_path = await self.fetch_and_save_job_details(job_id)
                    if job_details_path:
                        self.console.log(f"[green]Job {job_id} fetched successfully[/green]")
                        status_panel = Panel(f"Processing JobId: {job_id}", title="Status", border_style="yellow")
                        live.update(progress_table)

                        with open(job_details_path, 'r') as file:
                            job_details = json.load(file)
                        await self.process_job(job_details)

                        status_panel = Panel("Waiting for training jobs...", title="Status", border_style="green")
                        live.update(progress_table)
                        self.console.log(f"[green]Job {job_id} processed successfully[/green]")
                await asyncio.sleep(2)

if __name__ == "__main__":
    trainer = Trainer()
    signal.signal(signal.SIGINT, trainer.handle_interrupt)
    asyncio.run(trainer.main())

# import argparse
# import asyncio
# import json
# import os
# import signal
# import ssl
# import sys
# import time
# from communex.module.module import Module,endpoint
# import aiohttp
# import pyfiglet
# from dotenv import load_dotenv
# from rich.console import Console
# from rich.live import Live
# from rich.panel import Panel
# from rich.progress import Progress, SpinnerColumn, TextColumn
# from rich.table import Table
# from rich.text import Text

# # Append directories to sys.path for relative imports
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../', 'finetune')))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './', 'auth')))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'finetune/runpod')))
# from yogpt_subnet.miner.auth.auth import authenticate # type:ignore
# from yogpt_subnet.miner.finetune.gpt_fine_tune import fine_tune_gpt #type:ignore
# from yogpt_subnet.miner.utils.helpers import (fetch_and_save_job_details, fetch_jobs, register_completed_job, submit_to_runpod, update_job_status) #type:ignore
# from yogpt_subnet.miner.finetune.llama_fine_tune import fine_tune_llama # type:ignore
# from yogpt_subnet.miner.finetune.open_elm import fine_tune_openELM # type:ignore

# class Trainer(Module):
#     def __init__(self):
#         load_dotenv()
#         self.BASE_URL = os.getenv("BASE_URL")
#         self.HF_ACCESS_TOKEN = os.getenv("HF_ACCESS_TOKEN")
#         self.TOKEN = os.getenv("TOKEN")
#         self.MINER_ID = os.getenv("MINER_ID")
#         self.console = Console()
#         self.current_job_id = None  # Global variable to store the current job ID

#     @endpoint
#     def display_welcome_message(self):
#         fig = pyfiglet.Figlet(font='slant')
#         welcome_text = fig.renderText('YOGPT Miner')
#         self.console.print(welcome_text, style="bold blue")
#         self.console.print(Panel(Text("Welcome to YOGPT Miner System!", justify="center"), style="bold green"))

#     @endpoint
#     async def fetch_jobs(self):
#         timeout = aiohttp.ClientTimeout(total=120)  # Increased timeout to 120 seconds
#         ssl_context = ssl.create_default_context()
#         ssl_context.check_hostname = False
#         ssl_context.verify_mode = ssl.CERT_NONE

#         async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
#             headers = {'Authorization': f'Bearer {self.TOKEN}'}
#             try:
#                 async with session.get(f"{self.BASE_URL}/pending-jobs", headers=headers) as response:
#                     if response.status == 200:
#                         return await response.json()
#                     elif response.status == 401:
#                         return "Unauthorized"
#                     else:
#                         self.console.log(f"Failed to fetch jobs: {await response.text()}")
#                         return []
#             except aiohttp.ClientConnectorError as e:
#                 self.console.log(f"Connection error: {e}")
#                 return []
            
#     async def fetch_and_save_job_details(self, job_id):
#         async with aiohttp.ClientSession() as session:
#             headers = {'Authorization': f'Bearer {self.TOKEN}'}
#             async with session.post(f"{self.BASE_URL}/start-training/{job_id}", headers=headers, json={'minerId': self.MINER_ID}) as response:
#                 if response.status == 200:
#                     job_details = await response.json()
#                     job_dir = os.path.join(os.getcwd(), 'jobs', job_id)
#                     os.makedirs(job_dir, exist_ok=True)
#                     details_path = os.path.join(job_dir, 'details.json')
#                     job_details['jobId'] = job_id
#                     with open(details_path, 'w') as f:
#                         json.dump(job_details, f, indent=2)
#                     return details_path
#                 else:
#                     self.console.log(f"Failed to start training for job {job_id}: {await response.text()}")
#                     return None
 
#     async def update_job_status(self, job_id, status):
#         url = f"{self.BASE_URL}/update-status/{job_id}"
#         async with aiohttp.ClientSession() as session:
#             headers = {'Authorization': f'Bearer {self.TOKEN}', 'Content-Type': 'application/json'}
#             async with session.patch(url, json={'status': status}, headers=headers) as response:
#                 try:
#                     if response.status == 200:
#                         self.console.log(f"Status updated to {status} for job {job_id}")
#                     else:
#                         response.raise_for_status()
#                 except aiohttp.ClientResponseError as err:
#                     self.console.log(f"Failed to update status for job {job_id}: {err}")
#                 except Exception as e:
#                     self.console.log(f"An error occurred: {e}")
  
#     async def process_job(self, job_details, run_on_runpod=False, runpod_api_key=None):
#         global current_job_id
#         model_id = job_details['baseModel']
#         dataset_id = job_details['huggingFaceId']
#         job_id = job_details['jobId']
#         new_model_name = job_id
#         self.current_job_id = job_id  # Update the current job ID

#         try:
#             if run_on_runpod:
#                 if runpod_api_key is None:
#                     raise ValueError("RunPod API key is required when running on RunPod.")
#                 script_path = f"pipeline_script_{job_id}.py"
#                 generate_pipeline_script(job_details, script_path)
#                 submit_to_runpod(script_path, runpod_api_key)
#             else:
#                 model_repo_url = None
#                 self.console.log(f"General model "+model_id)
#                 model_detected = model_id.lower()
#                 if 'llama' in model_detected:
#                     self.console.log(f"model is  of type Llama:"+model_id)
#                     model_repo_url = await fine_tune_llama(model_id, dataset_id, new_model_name, self.HF_ACCESS_TOKEN, job_id)
#                 elif 'gpt' in model_detected:
#                     self.console.log(f"model is  of type GPT:"+model_id)
#                     model_repo_url = await fine_tune_gpt(model_id, dataset_id, new_model_name, self.HF_ACCESS_TOKEN, job_id)
#                 elif 'openelm' in model_detected:
#                     self.console.log(f"model is  of type OpenELM:"+model_id)
#                     model_repo_url = await fine_tune_openELM(model_id, dataset_id, new_model_name, self.HF_ACCESS_TOKEN, job_id)
#                 else:
#                     self.console.log(f"Unsupported model ID: {model_id}. Skipping job.")
                
#                     return  # Skip this job and proceed to the next one

#                 self.console.log(f"Model uploaded to: {model_repo_url}")

#             await self.update_job_status(job_id, 'completed')
#             await register_completed_job(job_id, model_repo_url)  # Pass the correct model_repo_url
#         except RuntimeError as e:
#             self.console.log(f"Failed to process job {job_id}: {str(e)}")
#         except Exception as e:
#             await self.update_job_status(job_id, 'pending')
#             self.console.log(f"Unexpected error occurred while processing job {job_id}: {str(e)}")
#         finally:
#             self.current_job_id = None  # Reset the current job ID
            
#     def handle_interrupt(self, signal, frame):
#         if self.current_job_id:
#             asyncio.run(self.update_job_status(self.current_job_id, 'pending'))
#         self.console.log("[bold red]Interrupted. Exiting...[/bold red]")
#         sys.exit(0)

#     async def main(self):
#         self.display_welcome_message()

#         username = input("Enter your username: ")
#         password = input("Enter your password: ")

#         token, miner_id = authenticate(username, password)
#         if not token or not miner_id:
#             self.console.log("[bold red]Authentication failed. Exiting...[/bold red]")
#             sys.exit(1)

#         self.TOKEN, self.MINER_ID = token, miner_id

#         progress_table = Table.grid(expand=True)
#         progress_table.add_column(justify="center", ratio=1)
#         status_panel = Panel("Waiting for training jobs...", title="Status", border_style="green")
#         progress_table.add_row(status_panel)

#         with Live(progress_table, refresh_per_second=10, console=self.console) as live:
#             while True:
#                 live.update(progress_table)
#                 jobs = await self.fetch_jobs()

#                 if jobs == "Unauthorized":
#                     self.console.log("[bold red]Unauthorized access. Please check your credentials.[/bold red]")
#                     break
#                 elif jobs:
#                     job_id = jobs[0]['id']
#                     self.console.log(f"Executing JobId: {job_id}")
#                     job_details_path = await self.fetch_and_save_job_details(job_id)
#                     if job_details_path:
#                         self.console.log(f"[green]Job {job_id} fetched successfully[/green]")
#                         status_panel = Panel(f"Processing JobId: {job_id}", title="Status", border_style="yellow")
#                         live.update(progress_table)

#                         with open(job_details_path, 'r') as file:
#                             job_details = json.load(file)
#                         await self.process_job(job_details)

#                         status_panel = Panel("Waiting for training jobs...", title="Status", border_style="green")
#                         live.update(progress_table)
#                         self.console.log(f"[green]Job {job_id} processed successfully[/green]")
#                 await asyncio.sleep(2)

# if __name__ == "__main__":
#     trainer = Trainer()
#     signal.signal(signal.SIGINT, trainer.handle_interrupt)
#     asyncio.run(trainer.main())
