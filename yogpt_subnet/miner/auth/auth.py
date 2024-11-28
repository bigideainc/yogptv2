import json
import os
import platform
import sys

import GPUtil
import psutil
import requests
from dotenv import find_dotenv, load_dotenv, set_key

# Load environment variables
load_dotenv()
BASE_URL = os.getenv("BASE_URL")

def get_system_details():
    # Get CPU and memory details
    memory = psutil.virtual_memory()
    storage = psutil.disk_usage('/')
    
    # Get GPU details using GPUtil
    gpus = GPUtil.getGPUs()
    gpu_info = []
    for gpu in gpus:
        gpu_info.append({
            "name": gpu.name,
            "total_memory": gpu.memoryTotal,
            "available_memory": gpu.memoryFree,
            "used_memory": gpu.memoryUsed
        })
    
    system_info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "cpu": platform.processor(),
        "cpu_count": psutil.cpu_count(logical=True),
        "total_memory": memory.total,            # Total RAM
        "available_memory": memory.available,    # Available RAM
        "used_memory": memory.used,              # Used RAM
        "total_storage": storage.total,          # Total Storage
        "available_storage": storage.free,       # Available Storage
        "used_storage": storage.used,            # Used Storage
        "gpus": gpu_info                         # GPU Information (list of GPUs)
    }
    
    return system_info

def authenticate(username, password):
    print("Username & password: ", username, " ", password)
    url = f"{BASE_URL}/login"
    
    system_details = get_system_details()
    
    response = requests.post(url, json={
        'username': username,
        'password': password,
        'system_details': system_details
    })
    
    if response.status_code == 200:
        token = response.json()['token']
        miner_id = response.json()['minerId']
        
        # Get the path to the .env file
        env_path = find_dotenv()
        
        # Update the .env file
        if env_path:
            set_key(env_path, "TOKEN", token)
            set_key(env_path, "MINER_ID", str(miner_id))
            print(f"Updated .env file at {env_path}")
            print("Authentication successful. Token and Miner ID saved.")
            return token, miner_id
        else:
            print("Could not find .env file.")
            return None, None
    else:
        print("Authentication failed.")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python auth.py command [args]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "login" and len(sys.argv) == 4:
        username = sys.argv[2]
        password = sys.argv[3]
        authenticate(username, password)
    else:
        print("Invalid command or arguments")

