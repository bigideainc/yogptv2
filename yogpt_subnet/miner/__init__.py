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
from yogpt_subnet.miner.trainer import Trainer  


class Miner(Trainer):
    def __init__(self, key: Keypair, settings: MinerSettings = None) -> None:
        super().__init__()
        self.settings = settings or MinerSettings()
        self.key = key
        self.c_client = CommuneClient(get_node_url(use_testnet=self.settings.use_testnet))
        self.netuid = get_netuid(self.c_client)
        self.stop_event = Event()
        # Print out key and UID information
        self.print_key_info()
        self.miner_uid= self.print_miner_uid()
    def print_key_info(self):
        """
        Prints out the key information (SS58 address, public key, etc.)
        on the network for the miner.
        """
        logger.info(f"Miner Key Information:")
        logger.info(f"SS58 Address: {self.key.ss58_address}")
        logger.info(f"Public Key (Hex): {self.key.public_key.hex()}")
        logger.info(f"Key Type: {self.key.crypto_type}")
        logger.info(f"Network UID: {self.netuid}")

    def print_miner_uid(self):
        """
        Prints out the miner's UID on the network based on its SS58 address.
        """
        try:
            # Get the map of keys (UID -> SS58 Address) for the network
            modules_keys = self.c_client.query_map_key(self.netuid)
            val_ss58 = self.key.ss58_address

            # Find the miner's UID by matching the SS58 address
            miner_uid = next(uid for uid, address in modules_keys.items() if address == val_ss58)

            logger.info(f"Miner UID on the network: {miner_uid}")
            return miner_uid

        except StopIteration:
            logger.error(f"Miner SS58 address {self.key.ss58_address} not found in the network.")
        except Exception as e:
            logger.error(f"Error retrieving miner UID: {e}")

    def serve(self):
        from communex.module.server import ModuleServer
        import uvicorn

        server = ModuleServer(self, self.key, subnets_whitelist=[self.netuid])
        app = server.get_fastapi_app()
        uvicorn.run(app, host=self.settings.host, port=self.settings.port)

if __name__ == "__main__":
    settings = MinerSettings(
        host="0.0.0.0",
        port=7777,
        model_type="llama2",
        job_id="003",
        dataset_id="mydataset",
        epochs=20,
        batch_size=64,
        learning_rate=2.0,
        hf_token="some_hf_token",
        use_testnet=True,
    )
    miner = Miner(key=classic_load_key("yogpt-miner0"), settings=settings)
    miner.serve(miner.run_pipeline(
        model_type=settings.model_type,
        dataset_id=settings.dataset_id,
        epochs=settings.epochs,
        batch_size=settings.batch_size,
        learning_rate=settings.learning_rate,
        hf_token=settings.hf_token,
        job_id=settings.job_id,
        miner_uid=miner.miner_uid
    ))
