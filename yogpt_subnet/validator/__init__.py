import asyncio
import threading
import time
import traceback
from datetime import datetime
from typing import List
from communex._common import get_node_url
from communex.client import CommuneClient
from communex.compat.key import classic_load_key
from communex.module.module import Module, endpoint
from loguru import logger
from pydantic import BaseModel
from substrateinterface import Keypair
from yogpt_subnet.base.utils import get_netuid
from yogpt_subnet.validator._config import ValidatorSettings
from yogpt_subnet.validator.validator_ import ModelRewardChecker

class WeightHistory(BaseModel):
    time: datetime
    data: List

class Validator(Module):
    def __init__(self, key: Keypair, settings: ValidatorSettings | None = None) -> None:
        super().__init__()
        self.settings = settings or ValidatorSettings()
        self.key = key
        self.client = self.c_client
        self.netuid = get_netuid(self.client)
        self.reward_checker = ModelRewardChecker(key=self.key, netuid=self.netuid, client=self.client)

    @property
    def c_client(self):
        return CommuneClient(get_node_url(use_testnet=self.settings.use_testnet))

    @endpoint
    def get_weights_history(self):
        return list(self.weights_histories)

    async def validate_step(self):
        await self.reward_checker.reward_completed_jobs()

    def validation_loop(self) -> None:
        settings = self.settings
        while True:
            try:
                logger.info(f"run validation loop")
                start_time = time.time()
                asyncio.run(self.validate_step())
                elapsed = time.time() - start_time
                if elapsed < settings.iteration_interval:
                    sleep_time = settings.iteration_interval - elapsed
                    logger.info(f"Sleeping for {sleep_time}")
                    time.sleep(sleep_time)
            except Exception as e:
                logger.error(f"Error in validation loop: {e}")
                logger.error(traceback.format_exc())

    def start_validation_loop(self):
        logger.info("start sync loop")
        self._loop_thread = threading.Thread(target=self.validation_loop, daemon=True)
        self._loop_thread.start()

    def serve(self):
        from communex.module.server import ModuleServer
        import uvicorn
        self.start_validation_loop()
        if self.settings.port:
            logger.info("server enabled")
            server = ModuleServer(self, self.key, subnets_whitelist=[self.netuid])
            app = server.get_fastapi_app()
            uvicorn.run(app, host=self.settings.host, port=self.settings.port)
        else:
            while True:
                time.sleep(60)

if __name__ == "__main__":
    settings = ValidatorSettings(use_testnet=True)
    Validator(key=classic_load_key("yogpt-validator0"), settings=settings).serve()