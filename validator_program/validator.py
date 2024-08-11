import asyncio
import concurrent.futures
import re
import time
from functools import partial
from communex.client import CommuneClient
from communex.module.client import ModuleClient
from communex.module.module import Module
from communex.types import Ss58Address
from substrateinterface import Keypair
import firebase_admin
from firebase_admin import credentials, firestore
from loguru import logger
import os
import warnings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

warnings.filterwarnings(
    "ignore",
    message="Detected filter using positional arguments. Prefer using the 'filter' keyword argument instead."
)

cred_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

if not os.path.exists(cred_path):
    raise FileNotFoundError(f"Credential file not found: {cred_path}")

# Regular expression to extract IP addresses
IP_REGEX = re.compile(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d+")

# Setup Firebase Admin
cred = credentials.Certificate(cred_path)
firebase_admin.initialize_app(cred)

# Initialize Firestore
db = firestore.client()

class TextValidator(Module):
    def __init__(self, key: Keypair, netuid: int, client: CommuneClient, call_timeout: int = 60):
        super().__init__()
        self.client = client
        self.key = key
        self.netuid = netuid
        self.call_timeout = call_timeout

    def get_addresses(self, netuid: int) -> dict[int, str]:
        """Retrieve module addresses in the subnet."""
        return self.client.query_map_address(netuid)

    def get_training_details(self, miner_info: tuple[list[str], Ss58Address]) -> dict:
        """
        Query the miner for training job details, including completion status, loss, duration, and miner ID.

        Args:
            miner_info: A tuple containing the miner's connection information and key.

        Returns:
            A dictionary containing training details such as 'completed', 'loss', 'duration', and 'miner_id'.
        """
        connection, miner_key = miner_info
        module_ip, module_port = connection
        client = ModuleClient(module_ip, int(module_port), self.key)

        try:
            training_details = asyncio.run(
                client.call(
                    "get_training_details",
                    miner_key,
                    {},
                    timeout=self.call_timeout,
                )
            )
            return training_details.get("details", {})
        except Exception as e:
            logger.error(f"Miner {module_ip}:{module_port} failed to provide training details: {e}")
            return {}

    def _score_miner(self, training_details: dict) -> float:
        """
        Score the miner based on training details using predefined criteria.

        Args:
            training_details: A dictionary containing 'completed', 'loss', and 'duration'.

        Returns:
            A score between 0 and 1 based on the miner's performance.
        """
        # Dummy criteria for now
        if not training_details.get("completed"):
            return 0.0

        loss = training_details.get("loss", float('inf'))
        duration = training_details.get("duration", "0:0:0")

        # Convert duration from HH:MM:SS to hours
        try:
            hours, minutes, seconds = map(int, duration.split(':'))
            duration_hours = hours + minutes / 60 + seconds / 3600
        except ValueError:
            logger.error(f"Invalid duration format: {duration}")
            return 0.0

        # Dummy scoring logic
        score = 0.0
        if loss < 0.3:
            score += 0.5
        if 5 <= duration_hours <= 10:
            score += 0.5

        return score

    def set_weights(self, score_dict: dict[int, float], settings) -> None:
        """Set weights based on miner scores and update Firestore."""
        score_dict = self.cut_to_max_allowed_weights(score_dict, settings.max_allowed_weights)
        scores = sum(score_dict.values())
        weighted_scores = {uid: int(score * 1000 / scores) for uid, score in score_dict.items() if score > 0}
        uids = list(weighted_scores.keys())
        weights = list(weighted_scores.values())
        
        # Vote on weights in the network
        self.client.vote(key=self.key, uids=uids, weights=weights, netuid=self.netuid)

        # Update Firestore with weights and miner IDs
        for uid, weight in weighted_scores.items():
            self.log_miner_weight(uid, weight)

    def log_miner_weight(self, miner_id: int, weight: int) -> None:
        """Log miner weight in Firestore."""
        try:
            miner_ref = db.collection('miners').document(str(miner_id))
            miner_data = {
                'weight': weight,
                'timestamp': firestore.SERVER_TIMESTAMP  # Automatically set the current time
            }
            miner_ref.set(miner_data, merge=True)  # Merge with existing data if any
            logger.info(f"Logged weight for miner {miner_id}: {weight}")
        except Exception as e:
            logger.error(f"Failed to log weight for miner {miner_id}: {e}")

    def cut_to_max_allowed_weights(self, score_dict: dict[int, float], max_allowed_weights: int) -> dict[int, float]:
        """Limit scores to the maximum allowed weights."""
        sorted_scores = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_scores[:max_allowed_weights])

    async def validate_step(self, syntia_netuid: int, settings) -> None:
        """Perform a validation step by interacting with miner modules."""
        modules_adresses = self.get_addresses(syntia_netuid)
        modules_keys = self.client.query_map_key(syntia_netuid)
        modules_info = {
            module_id: (self.get_ip_port(modules_adresses[module_id]), modules_keys[module_id])
            for module_id in modules_keys
        }

        score_dict = {}

        for uid, miner_info in modules_info.items():
            training_details = self.get_training_details(miner_info)
            miner_id = training_details.get("miner_id", uid)
            score = self._score_miner(training_details)
            if score > 0:
                score_dict[miner_id] = score

        if score_dict:
            self.set_weights(score_dict, settings)

    def validation_loop(self, settings) -> None:
        """Continuously perform validation steps."""
        while True:
            start_time = time.time()
            asyncio.run(self.validate_step(self.netuid, settings))
            elapsed = time.time() - start_time
            if elapsed < settings.iteration_interval:
                time.sleep(settings.iteration_interval - elapsed)

    def get_ip_port(self, module_address: str):
        """Extract IP and port from module address."""
        match = re.search(IP_REGEX, module_address)
        return match.group(0).split(":") if match else (None, None)

# Settings class with max_allowed_weights and iteration_interval
class ValidatorSettings:
    def __init__(self, max_allowed_weights: int, iteration_interval: float):
        self.max_allowed_weights = max_allowed_weights
        self.iteration_interval = iteration_interval