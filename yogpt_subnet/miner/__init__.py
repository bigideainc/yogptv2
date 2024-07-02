from communex.module.module import Module
from communex.client import CommuneClient
from communex.module.client import ModuleClient
from communex.compat.key import classic_load_key
from communex.types import Ss58Address
from substrateinterface import Keypair
from communex.key import generate_keypair
from communex._common import get_node_url

from yogpt_subnet.miner._config import MinerSettings # type:ignore
from yogpt_subnet.base.utils import get_netuid
from yogpt_subnet.miner.auth.trainer import Trainer  # type:ignore

import sys
import threading
import signal
import asyncio
import uvicorn
import argparse
from loguru import logger


class Miner(Module):
    def __init__(self, key: Keypair, settings: MinerSettings = None) -> None:
        super().__init__()
        self.settings = settings or MinerSettings()
        self.key = key
        self.c_client = CommuneClient(
            get_node_url(use_testnet=self.settings.use_testnet)
        )
        self.netuid = get_netuid(self.c_client)
        self.trainer = Trainer()

    def serve(self):
        from communex.module.server import ModuleServer

        server = ModuleServer(self, self.key, subnets_whitelist=[self.netuid])
        app = server.get_fastapi_app()

        # Start the trainer in a new thread
        trainer_thread = threading.Thread(target=self.run_trainer, args=())
        trainer_thread.start()

        uvicorn.run(app, host=self.settings.host, port=self.settings.port)

    def run_trainer(self):
        parser = argparse.ArgumentParser(description="Automated training and uploading")
        parser.add_argument('--wallet_address', type=str, required=True)
        parser.add_argument('--runpod', action='store_true', help="Run the job on RunPod")
        parser.add_argument('--runpod_api_key', type=str, help="RunPod API key")
        args = parser.parse_args()

        signal.signal(signal.SIGINT, self.trainer.handle_interrupt)
        asyncio.run(self.trainer.main(args))


if __name__ == "__main__":
    settings = MinerSettings(
        host="0.0.0.0",
        port=7777,
        use_testnet=True,
    )
    miner = Miner(key=classic_load_key("yogpt-miner0"), settings=settings)
    miner.serve()
