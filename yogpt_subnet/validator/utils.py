import asyncio
import json
import os
import websockets
import sys
import time
import aiohttp
import requests
from typing import List, Dict, Optional
from dotenv import load_dotenv    
from huggingface_hub import HfApi, Repository
from datetime import datetime
from huggingface_hub import HfApi, Repository,hf_hub_download



async def fetch_open_jobs() -> List[str]:
    """
    Fetch open jobs via WebSocket connection with proper error handling and timeouts.
    
    Returns:
        List[str]: List of open job IDs
    """
    load_dotenv()
    base_url = os.getenv('BASE_URL')
    if not base_url:
        print("BASE_URL not set in environment")
        return []

    websocket_url = f"{base_url}/ws/jobs/open"
    open_jobs_list = []
    
    try:
        async with websockets.connect(
            websocket_url,
            ping_timeout=30,
            close_timeout=10,
            max_size=10_485_760  # 10MB max message size
        ) as websocket:
            try:
                # Use asyncio.wait_for to set a timeout
                message = await asyncio.wait_for(websocket.recv(), timeout=5)
                data = json.loads(message)
                
                if "open_jobs" in data:
                    open_jobs = data["open_jobs"]
                    open_jobs_list = [
                        job["job_id"] 
                        for job in open_jobs 
                        if job.get("status") == "open"
                    ]
                elif "error" in data:
                    print(f"Server error: {data['error']}")
                else:
                    print(f"Unexpected message format: {data}")
                        
            except asyncio.TimeoutError:
                print("Timeout waiting for server response")
                
    except websockets.exceptions.InvalidURI:
        print(f"Invalid WebSocket URL: {websocket_url}")
    except websockets.exceptions.ConnectionClosed as e:
        print(f"WebSocket connection closed: {e}")
    except Exception as e:
        print(f"Connection error: {str(e)}")
        
    return open_jobs_list

def update_job_status(job_id: str):
    """
    Update the status of a job using a REST API call.

    Args:
        job_id (str): The ID of the job to update.

    Returns:
        dict: Response JSON or error details.
    """
    new_status = "Closed"
    base_url_mode = "https://a9labsapi-1048667232204.us-central1.run.app"
    url = f"{base_url_mode}/jobs/update"

    try:
        response = requests.post(url, data={"status": new_status,"job_id":job_id})
        print("Response status code:", response.status_code)
        print("Response text:", response.text)

        response.raise_for_status()  # Raise error for HTTP 4xx/5xx
        print(f"Job {job_id} status updated to '{new_status}' successfully.")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Failed to update job status: {str(e)}")
        return {"error": str(e)}



def fetch_training_metrics_commits(repo_id: str) -> List[Dict]:
    try:
        token = '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'
        api = HfApi(token=token)
        model_repo_url = f"https://huggingface.co/{repo_id}"
        commits = api.list_repo_commits(repo_id=repo_id, token=token)

        training_metrics = []
        processed_commits = 0

        print(f"Found {len(commits)} total commits in repository")

        for commit in commits:
            try:
                files = api.list_repo_tree(repo_id=repo_id, revision=commit.commit_id, token=token)
                json_files = [f for f in files if f.path.endswith('.json')]

                for json_file in json_files:
                    try:
                        local_path = hf_hub_download(repo_id=repo_id, filename=json_file.path, 
                                                      revision=commit.commit_id, token=token)

                        with open(local_path, 'r') as f:
                            content = f.read()
                            metrics_data = json.loads(content)

                        # Ensure 'metrics' and 'miner_uid' exist in the JSON data
                        if isinstance(metrics_data, dict) and "metrics" in metrics_data:
                            miner_uid = metrics_data.get("miner_uid")
                            job_id = metrics_data["metrics"].get("job_id") if "metrics" in metrics_data else None

                            # Create metrics entry only if miner_uid is present and job_id is found in metrics
                            if miner_uid and job_id:
                                metrics_entry = {
                                    "model_repo": metrics_data.get("model_repo", "unknown"),
                                    "metrics": metrics_data["metrics"],
                                    "miner_uid": miner_uid,
                                    "job_id": job_id,
                                    "timestamp": metrics_data.get("timestamp", "unknown")
                                }

                                training_metrics.append(metrics_entry)
                                processed_commits += 1

                    except json.JSONDecodeError:
                        print(f"Could not decode JSON in file: {json_file.path}")
                        continue

            except Exception as e:
                print(f"Error processing commit {commit.commit_id}: {str(e)}")
                continue

        # Filter out entries that have both miner_uid and job_id
        filtered_metrics = [entry for entry in training_metrics if entry.get('miner_uid') and entry['metrics'].get('job_id')]

        print(f"Successfully processed {processed_commits} commits with valid metrics")
        return filtered_metrics

    except Exception as e:
        print(f"Error fetching commits: {str(e)}")
        return []