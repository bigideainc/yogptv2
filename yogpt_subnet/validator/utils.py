# import asyncio
# import json
# import os
# import aiohttp
# from dotenv import load_dotenv
# from datetime import datetime, timedelta

# load_dotenv()
# BASE_URL = os.getenv("BASE_URL")
# async def fetch_completed_jobs():
#     print("Fetching completed jobs")
#     async with aiohttp.ClientSession() as session:
#         async with session.get(f"{BASE_URL}/completed-jobs") as response:
#             if response.status == 200:
#                 jobs = await response.json()
#                 return [
#                     {
#                         "id": job["id"],
#                         "jobId": job["jobId"],
#                         "loss": float(job["loss"]),
#                         "accuracy": float(job["accuracy"]),
#                         "huggingFaceRepoId": job["huggingFaceRepoId"],
#                         "totalPipelineTime": job["totalPipelineTime"],
#                         "minerId": job["minerId"],
#                         "completedAt": job["completedAt"],
#                         "reward": float(job.get("reward", 0)),
#                         "reward_message": job["reward_message"],
#                         "model_tuned": job.get("model_tuned", None)
#                     }
#                     for job in jobs
#                     if "status" not in job
#                 ]
#             elif response.status == 401:
#                 return "Unauthorized"
#             else:
#                 print(f"Failed to fetch jobs: {await response.text()}")
#                 return []