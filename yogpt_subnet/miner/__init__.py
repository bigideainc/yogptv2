import sys
import threading
import signal
import asyncio
import uvicorn
from loguru import logger
from multiprocessing import Event

from communex.module.module import Module
from communex.client import CommuneClient
from communex.module.client import ModuleClient
from communex.compat.key import classic_load_key
from communex.types import Ss58Address
from substrateinterface import Keypair
from communex.key import generate_keypair
from communex._common import get_node_url

from yogpt_subnet.miner._config import MinerSettings  # type: ignore
from yogpt_subnet.base.utils import get_netuid
from yogpt_subnet.miner.auth.trainer import Trainer  # type: ignore


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
        self.stop_event = Event()

    def serve(self):
        from communex.module.server import ModuleServer

        server = ModuleServer(self, self.key, subnets_whitelist=[self.netuid])
        app = server.get_fastapi_app()

        # Start the trainer in a new thread
        trainer_thread = threading.Thread(target=self.run_trainer)
        trainer_thread.start()

        uvicorn.run(app, host=self.settings.host, port=self.settings.port)

    def run_trainer(self):
        asyncio.run(self.trainer_main())

    async def trainer_main(self):
        while not self.stop_event.is_set():
            try:
                await self.trainer.main()
            except asyncio.CancelledError:
                break

    def handle_interrupt(self, signum, frame):
        logger.info("Interrupt received, stopping trainer...")
        self.stop_event.set()
        self.trainer.stop()  # Ensure trainer has a stop method to cleanup resources

if __name__ == "__main__":
    settings = MinerSettings(
        host="0.0.0.0",
        port=7777,
        use_testnet=True,
    )
    miner = Miner(key=classic_load_key("yogpt-miner0"), settings=settings)

    # Set up signal handling in the main thread
    signal.signal(signal.SIGINT, miner.handle_interrupt)
    signal.signal(signal.SIGTERM, miner.handle_interrupt)

    miner.serve()
