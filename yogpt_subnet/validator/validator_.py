import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime, timedelta
from communex.module.module import Module
from communex.module.module import Module
from communex.client import CommuneClient
from communex.types import Ss58Address  
from substrateinterface import Keypair
from loguru import logger
import os
import time
import warnings
from dotenv import load_dotenv

warnings.filterwarnings(
    "ignore",
    message="Detected filter using positional arguments. Prefer using the 'filter' keyword argument instead."
)

load_dotenv()

cred_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

if not os.path.exists(cred_path):
    raise FileNotFoundError(f"Credential file not found: {cred_path}")

cred = credentials.Certificate(cred_path)
firebase_admin.initialize_app(cred)

db = firestore.client()

class ModelRewardChecker(Module):
    def __init__(self,key: Keypair, netuid: int, client: CommuneClient) -> None:
        super().__init__()
        self.key = key
        self.netuid = netuid
        self.client = client
        self.model_thresholds = {
            "llama2-7b": {"threshold": 0.20, "training_per_hour": 1.2, "fine_tuning_time": (10, 12)},
            "OpenELM-270M": {"threshold": 0.50, "training_per_hour": 1.0, "fine_tuning_time": (3, 5)},
            "OpenELM-450M": {"threshold": 0.35, "training_per_hour": 1.3, "fine_tuning_time": (6, 8)},
            "OpenELM-3B": {"threshold": 0.20, "training_per_hour": 2.2, "fine_tuning_time": (10, 12)},
            "GPT2": {"threshold": 0.50, "training_per_hour": 1.5, "fine_tuning_time": (3, 5)},
            "LLama3B": {"threshold": 0.35, "training_per_hour": 2.2, "fine_tuning_time": (6, 8)}
        }

    def calculate_reward(self, job_data):
        model_tuned = job_data.get('model_tuned')
        loss = job_data.get('loss')
        duration_str = job_data.get('totalPipelineTime')
        model_created = job_data.get('huggingFaceRepoId')
        
        if not model_created or 'huggingface' not in model_created.lower():
            return 0, "No valid Hugging Face model created"

        if model_tuned not in self.model_thresholds:
            return 0, "Model not found in thresholds"

        model_info = self.model_thresholds[model_tuned]
        threshold = model_info['threshold']
        training_per_hour = model_info['training_per_hour']
        min_time, max_time = model_info['fine_tuning_time']

        try:
            duration_parts = duration_str.split(':')
            if len(duration_parts) == 3:  # HH:MM:SS format
                hours, minutes, seconds = map(int, duration_parts)
                duration = hours + minutes / 60 + seconds / 3600
            elif len(duration_parts) == 2:  # HH:MM format
                hours, minutes = map(int, duration_parts)
                duration = hours + minutes / 60
            else:
                return 0, f"Invalid duration format: {duration_str}"
        except ValueError:
            return 0, f"Unable to parse duration: {duration_str}"

        if loss > threshold:
            return 0, "Loss exceeds threshold"

        if duration < min_time:
            return 0, f"Training completed too quickly. Expected minimum {min_time} hours, but took {duration:.2f} hours"

        if duration > max_time:
            return 0, f"Training took too long. Expected maximum {max_time} hours, but took {duration:.2f} hours"

        reward = training_per_hour * duration
        return reward, f"Reward granted for {duration:.2f} hours of training"
    
    def assign_weight(self, score):
        """
        Assign a weight based on the score. The score is normalized into a weight.
        """
        max_score = 1.0 
        weight = int(score * 1000 / max_score)
        return weight

    def reward_completed_jobs(self):
        jobs_ref = db.collection('completed_jobs').where('status', '==', 'pending_reward')
        completed_jobs = jobs_ref.stream()

        score_dict = {}
        for job in completed_jobs:
            job_data = job.to_dict()
            reward, message = self.calculate_reward(job_data)
            logger.info(f"{reward} {message}")
            
            if reward > 0:
                # Assign weight based on reward (score)
                score = reward / 100  # Normalize reward to a score out of 1
                weight = self.assign_weight(score)
                score_dict[job_data['minerId']] = score

                job.reference.update({
                    'status': 'rewarded',
                    'reward': reward,
                    'reward_message': message,
                    'weight': weight
                })
                self.update_miner_account(job_data['minerId'], reward)
            else:
                # Update the job status with no reward
                job.reference.update({
                    'status': 'not_rewarded',
                    'reward_message': message
                })

        if score_dict:
            self.set_weights(score_dict)
    def set_weights(self, score_dict: dict[int, float]) -> None:
        """
        Set weights for miners based on their scores and update the blockchain.
        """
        # Apply cutting logic to conform with subnet weight limits
        score_dict = self.cut_to_max_allowed_weights(score_dict)

        weighted_scores = {uid: self.assign_weight(score) for uid, score in score_dict.items()}
        uids = list(weighted_scores.keys())
        weights = list(weighted_scores.values())

        # Call the blockchain to vote and set weights
        self.client.vote(key=self.key, uids=uids, weights=weights, netuid=self.netuid)
    
    def cut_to_max_allowed_weights(self, score_dict: dict[int, float], max_allowed_weights: int = 420) -> dict[int, float]:
        """
        Cut the scores to the maximum allowed weights.
        """
        sorted_scores = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
        cut_scores = sorted_scores[:max_allowed_weights]
        return dict(cut_scores)

    def update_miner_account(self, miner_id, reward):
        miner_ref = db.collection('miners').document(miner_id)
        miner_data = miner_ref.get().to_dict()

        if miner_data is None:
            logger.error(f"Miner with ID {miner_id} not found")
            return

        new_balance = miner_data.get('Account', 0) + reward
        miner_ref.update({'Account': new_balance})

        logger.info(f"Updated miner {miner_id} account. New balance: {new_balance}")

    
    def validation_loop(self):
        """
        Run the validation loop continuously based on the provided settings.
        """
        while True:
            start_time = time.time()
            self.reward_completed_jobs()
            elapsed = time.time() - start_time

            if elapsed < self.settings.iteration_interval:
                sleep_time = self.settings.iteration_interval - elapsed
                logger.info(f"Sleeping for {sleep_time} seconds")
                time.sleep(sleep_time)

if __name__ == "__main__":
    key = Keypair.create_from_uri('//ValidatorKey')
    netuid = 12
    client = CommuneClient('wss://your-blockchain-node')

    validator = ModelRewardChecker(key=key, netuid=netuid, client=client)
    validator.validation_loop()