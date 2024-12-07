from communex.module.module import Module
from communex.client import CommuneClient
from substrateinterface import Keypair
from huggingface_hub import HfApi
from yogpt_subnet.validator.utils import fetch_training_metrics_commits,fetch_open_jobs,update_job_status
from loguru import logger
import math
import warnings
import sys
import websockets
from dotenv import load_dotenv
warnings.filterwarnings("ignore",message="Detected filter using positional arguments. Prefer using the 'filter' keyword argument instead.")

load_dotenv()

def sigmoid(x:float):
    return 1/(1+math.exp(-x))

class ModelRewardChecker(Module):
    def __init__(self, key: Keypair, netuid: int, client: CommuneClient) -> None:
        super().__init__()
        self.key = key
        self.netuid = netuid
        self.client = client
        self.repo_name="Tobius/soonish"
        logger.info(f"Model reward checker initialized")

    def read_commits(self):
        """Read commits from the central Hugging Face repository."""
        commits = fetch_training_metrics_commits(repo_id=self.repo_name)
        return commits

    async def group_commits(self, commits):
        """Group commits by job."""
        print(commits)
        job_groups = {}
        result = await fetch_open_jobs()
        print(f"available jobs {result}")
        for commit in commits:
            job_id = commit["metrics"]["job_id"]
            if job_id in result:
                if job_id not in job_groups:
                    job_groups[job_id] = []
                job_groups[job_id].append(commit)
        return job_groups
    
    def extract_metrics_by_job_id(self, job_id, commits):
        # Initialize an empty list to store the results
        results = []
        
        # Iterate over each commit in the commits list
        for commit in commits:
            # Check if the 'job_id' in the commit matches the input job_id
            if commit['job_id'] == job_id:
                # Extract the needed information
                miner_uid = commit['miner_uid']
                final_loss = commit['metrics']['final_loss']
                model_repo = commit['model_repo']
                
                # Append the extracted data to the results list
                results.append({
                    'miner_uid': miner_uid,
                    'final_loss': final_loss,
                    'model_repo': model_repo
                })
        return results
    
    def score_miners(self, metrics_list):
        """
        Scores miners based on their final_loss, rewards the best miner, and ranks all miners.

        Parameters:
            metrics_list (list): A list of dictionaries containing 'miner_uid', 'final_loss', and 'model_repo'.

        Returns:
            dict: A dictionary containing:
                - 'ranked_miners': List of miners with their ranking positions, 'miner_uid', and 'final_loss'.
                - 'best_miner': Details of the best miner with 'miner_uid', 'final_loss', and 'model_repo'.
                - 'rewards': A dictionary mapping 'miner_uid' to their reward (1 token for the best miner).
        """
        # Sort the miners based on their final_loss in ascending order
        sorted_miners = sorted(metrics_list, key=lambda x: x['final_loss'])
        
        # Assign positions (rankings) to the miners
        ranked_miners = []
        for position, miner in enumerate(sorted_miners, start=1):
            ranked_miners.append({
                'position': position,
                'miner_uid': miner['miner_uid'],
                'final_loss': miner['final_loss']
            })
        
        # Identify the best miner (the one with the lowest final_loss)
        best_miner = sorted_miners[0]
        
        # Award one token to the best miner
        rewards = {best_miner['miner_uid']: 1.0}
        
        # Extract details of the best miner
        best_miner_info = {
            'miner_uid': best_miner['miner_uid'],
            'final_loss': best_miner['final_loss'],
            'model_repo': best_miner['model_repo']
        }
        
        # Return the results
        return {
            'ranked_miners': ranked_miners,
            'best_miner': best_miner_info,
            'rewards': rewards
        }

    async def reward_completed_jobs(self):
        logger.info("Checking completed jobs to evaluate ...")
        try:
            commits = self.read_commits()
        except Exception as e:
            logger.error(f"Failed to read commits: {e}")
            return
    
        logger.info("Grouping jobs ...")
        try:
            job_groups = await self.group_commits(commits)
        except Exception as e:
            logger.error(f"Failed to group jobs: {e}")
            return

        for job_id, commits in job_groups.items():
            metrics_list = self.extract_metrics_by_job_id(job_id, commits)
            if metrics_list:
                print(f"rewarding and scoring miners for jobid {job_id}")
                results = self.score_miners(metrics_list)
                print(f"job results {results}")
                score_dict = {}
                for miner_uid, score in results['rewards'].items():
                    score_dict[miner_uid] = score
                self.set_weights_jobupdate(score_dict, job_id)
        
    def set_weights_jobupdate(self,score_dict: dict[str, float], job_id: str):
        if score_dict:
            self.set_weights(score_dict)
            update_job_status(job_id)

    def set_weights(self, score_dict: dict[str, float]) -> None:
        logger.info(f"Setting weights for miners: {score_dict}")

        score_dict = self.cut_to_max_allowed_weights(score_dict)

        modules_keys = self.client.query_map_key(self.netuid)
        uid_scores = {}
        for ss58_address, score in score_dict.items():
            miner_uid = next((uid for uid, address in modules_keys.items() if address == ss58_address), None)
            
            if miner_uid is not None:
                uid_scores[miner_uid] = score
            else:
                logger.warning(f"SS58 address {ss58_address} not found in network, skipping.")
        if not uid_scores:
            logger.error("No valid UIDs were found for the provided SS58 addresses.")
            return
        weighted_scores = {uid: self.assign_weight(score) for uid, score in uid_scores.items()}
        uids = list(weighted_scores.keys())
        weights = list(weighted_scores.values())
        try:
            self.client.vote(key=self.key, uids=uids, weights=weights, netuid=self.netuid)
        except Exception as e:
            logger.error(f"Error setting weights: {e}")

    def assign_weight(self, score):
        max_score = 1.0 
        weight = int(score * 5000 / max_score)
        return weight
    def cut_to_max_allowed_weights(self, score_dict: dict[int, float], max_allowed_weights: int = 420) -> dict[int, float]:
        if len(score_dict) > max_allowed_weights:
            sorted_scores = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
            cut_scores = sorted_scores[:max_allowed_weights]
            logger.info(f"Scores after cutting to max allowed weights: {cut_scores}")
            return dict(cut_scores)
        return score_dict
