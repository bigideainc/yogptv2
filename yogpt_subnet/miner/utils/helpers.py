import asyncio
import json
import os
import sys
import time

import aiohttp
import requests
import runpod
from dotenv import load_dotenv

load_dotenv()
BASE_URL = os.getenv("BASE_URL")
HF_ACCESS_TOKEN = os.getenv("HF_ACCESS_TOKEN")
TOKEN = os.getenv("TOKEN")
MINER_ID = os.getenv("MINER_ID")

# # current file directory
# root = os.path.dirname(os.path.abspath(__file__))


def create_runpod_instance(model_id):
    gpu_count = 1
    pod = runpod.create_pod(
        name="Apple_train",
        image_name="runpod/stable-diffusion",
        gpu_type_id="NVIDIA GeForce RTX 4090",
        data_center_id="EU-RO-1",
        cloud_type="SECURE",
        docker_args=f"--model-id {model_id}",
        gpu_count=gpu_count,
        volume_in_gb=5,
        container_disk_in_gb=5,
        ports="8080/http,29500/http",
        volume_mount_path="/data",
    )
    
    if pod:
        print(f"Pod created successfully: {pod}")
        return pod["pod_id"]
    else:
        print(f"Failed to create pod")
        return None

def wait_for_pod_ready(pod_id):
    while True:
        status = runpod.get_pod_status(pod_id)
        if status == "RUNNING":
            print(f"Pod {pod_id} is ready")
            break
        else:
            print(f"Pod {pod_id} status: {status}")
        time.sleep(10)

def submit_job_to_pod(script_path, pod_id):
    job = runpod.submit_job(pod_id, script_path, command=f"python {os.path.basename(script_path)}")
    if job:
        print(f"Job submitted successfully: {job}")
    else:
        print(f"Failed to submit job")

def submit_to_runpod(script_path, runpod_api_key):
    runpod.api_key = runpod_api_key  # Set the API key for runpod
    model_id = "apple/OpenELM-450M"  # Hard-coded model ID for example purposes
    pod_id = create_runpod_instance(model_id)
    if pod_id:
        wait_for_pod_ready(pod_id)
        submit_job_to_pod(script_path, pod_id)

async def fetch_jobs():
    print("Waiting for training jobs")
    async with aiohttp.ClientSession() as session:
        headers = {'Authorization': f'Bearer {TOKEN}'}
        async with session.get(f"{BASE_URL}/pending-jobs", headers=headers) as response:
            if response.status == 200:
                return await response.json()
            elif response.status == 401:
                return "Unauthorized"
            else:
                print(f"Failed to fetch jobs: {await response.text()}")
                return []

async def fetch_and_save_job_details(job_id):
    async with aiohttp.ClientSession() as session:
        headers = {'Authorization': f'Bearer {TOKEN}'}
        async with session.post(f"{BASE_URL}/start-training/{job_id}", headers=headers,
                                json={'minerId': MINER_ID}) as response:
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
                print(f"Failed to start training for job {job_id}: {await response.text()}")
                return None
                
async def update_job_status(job_id, status):
    url = f"{BASE_URL}/update-status/{job_id}"
    async with aiohttp.ClientSession() as session:
        headers = {'Authorization': f'Bearer {TOKEN}', 'Content-Type': 'application/json'}
        async with session.patch(url, json={'status': status}, headers=headers) as response:
            try:
                if response.status == 200:
                    print(f"Status updated to {status} for job {job_id}")
                else:
                    response.raise_for_status()
            except aiohttp.ClientResponseError as err:
                print(f"Failed to update status for job {job_id}: {err}")
            except Exception as e:
                print(f"An error occurred: {e}")
                                
async def register_completed_job(job_id, huggingFaceRepoId, loss, accuracy, total_pipeline_time):
    url = f"{BASE_URL}/complete-training"
    async with aiohttp.ClientSession() as session:
        headers = {'Authorization': f'Bearer {TOKEN}', 'Content-Type': 'application/json'}
        payload = {
            'jobId': job_id,
            'huggingFaceRepoId': huggingFaceRepoId,
            'loss': loss,
            "accuracy": accuracy,
            'minerId': MINER_ID,
            'totalPipelineTime': total_pipeline_time
        }
        async with session.post(url, json=payload, headers=headers) as response:
            try:
                if response.status == 200:
                    print(f"Completed job registered successfully for job {job_id}")
                else:
                    response.raise_for_status()
            except aiohttp.ClientResponseError as err:
                print(f"Failed to register completed job {job_id}: {err}")
            except Exception as e:
                print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(fetch_jobs())
