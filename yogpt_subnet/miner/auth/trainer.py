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
from communex.client import CommuneClient
from communex._common import get_node_url
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from yogpt_subnet.miner.finetune.gpt_fine_tune import \
    fine_tune_gpt  # type:ignore
from yogpt_subnet.miner.finetune.llama_fine_tune import \
    fine_tune_llama  # type:ignore
from yogpt_subnet.miner.finetune.open_elm import \
    fine_tune_openELM  # type:ignore
from yogpt_subnet.miner.finetune.gemma_fine_tune import fine_tune_gemma  #type: ignore

class Trainer(Module):
    def __init__(self,netuid):
        super().__init__()
        self.netuid = netuid
        self.console = Console()

    @endpoint
    def display_welcome_message(self):
        fig = pyfiglet.Figlet(font='slant')
        welcome_text = fig.renderText('A9Labs Commune')
        self.console.print(welcome_text, style="bold blue")

    async def run_pipeline(self, model_type, dataset_id, epochs, batch_size, learning_rate, hf_token, job_id):
        """
        Dynamically select and run the appropriate pipeline based on model_type.
        """
        pipelines = {
            "llama2": fine_tune_llama,
            # "gpt2": fine_tune_gpt
        }

        if model_type not in pipelines:
            self.console.log(f"[red]Unsupported model type: {model_type}. Available options: {list(pipelines.keys())}[/red]")
            return

        self.console.log(f"[blue]Starting pipeline for model type: {model_type}[/blue]")
        try:
            pipeline_function = pipelines[model_type]
            model_repo_url, loss, accuracy, total_pipeline_time = await pipeline_function(
                dataset_id=dataset_id,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                hf_token=hf_token,
                job_id=job_id,
                miner_uid=self.netuid
            )
            if model_repo_url:
                self.console.log(f"[green]Model uploaded to: {model_repo_url}[/green]")
                self.console.log(f"[green]Total pipeline time: {total_pipeline_time}[/green]")
            else:
                self.console.log(f"[red]Pipeline for Job ID {job_id} did not return a model URL[/red]")
        except Exception as e:
            self.console.log(f"[red]Error while running pipeline for Job ID {job_id}: {str(e)}[/red]")
 

    async def main(self, args):
        """Main execution function."""
        self.display_welcome_message()
        progress_table = Table.grid(expand=True)
        progress_table.add_column(justify="center", ratio=1)
        status_panel = Panel("Initializing pipeline...", title="Status", border_style="yellow")
        progress_table.add_row(status_panel)

        with Live(progress_table, refresh_per_second=10, console=self.console) as live:
            live.update(progress_table)

            # Run the selected pipeline
            await self.run_pipeline(
                model_type=args.model_type,
                dataset_id=args.dataset_id,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                hf_token=args.hf_token,
                job_id=args.job_id
            )

            status_panel = Panel("Pipeline completed.", title="Status", border_style="green")
            live.update(progress_table)
            self.console.log(f"[green]Pipeline completed successfully.[/green]")

def get_netuid_automatically():
    """Retrieve netuid from the CommuneClient."""
    client = CommuneClient(get_node_url(use_testnet=True))
    return get_netuid(client)

if __name__ == "__main__":
    netuid = get_netuid_automatically()
    parser = argparse.ArgumentParser(description="Run a fine-tuning pipeline.")
    parser.add_argument("--model_type", type=str, required=True, help="Type of model to fine-tune (e.g., llama2, gpt2).")
    parser.add_argument("--job_id", type=str, required=True, help="Job indentification.")
    parser.add_argument("--dataset_id", type=str, required=True, help="Dataset ID on Hugging Face.")
    parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, required=True, help="Learning rate for training.")
    parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face access token.")
    args = parser.parse_args()
    trainer = Trainer(netuid=netuid)
    asyncio.run(trainer.main(args))

