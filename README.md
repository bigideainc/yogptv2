
---
title: "A9Labs Commune Subnet Setup Guide"
output: html_document
---

# Overview
The A9Labs Commune Subnet provides an environment for fine-tuning large language models (LLMs) on custom datasets. Participants (miners) train models while validators evaluate submissions, awarding the miner with the lowest evaluation loss.

## Prerequisites
Ensure you have the following installed:
- Python 3.8 or higher
- pip
- Git
- curl
- Atleast 12GB RAM
- Atleast 1 GPU

## Installation
```{r, eval=FALSE}
# Clone repository
system("git clone https://github.com/tobiusaolo/yogptv2.git")

# Navigate to project directory
setwd("yogptv2")
```

## Project Setup
```{r, eval=FALSE}
# Install dependencies using Poetry
system("poetry install")

# Enter the Poetry environment
system("poetry shell")

# Install additional requirements
system("pip install -r requirements.txt")

# Install TRL package
system("pip install git+https://github.com/huggingface/trl.git")
```

## Wallet Setup
Create a new wallet key:
```{r, eval=FALSE}
system("comx key create <your-key-name>")
```

## Registering as a Miner or Validator
To register as a miner or validator, you must have at least 10 COMAI tokens:
```{r, eval=FALSE}
# Register as a miner
system("comx module register miner <your-key-name> 12")

# Register as a validator
system("comx module register validator <your-key-name> 12")
```

## Running a Miner Node
Visit the [A9Labs Dashboard](https://tobiusaolo.github.io/A9labsDashboard/). Select a job and note down the `job_id` and `dataset_id`. Run a miner node:
```{r, eval=FALSE}
system("python yogpt_subnet/cli.py miner <your-key-name> <ip-address> <port> <model-type> <job-id> <dataset-id> <epochs> <batch-size> <learning-rate> <your-hf-token>")
```

## Running a Validator Node
Run a validator node:
```{r, eval=FALSE}
system("python yogpt_subnet/cli.py validator <your-key-name> <ip-address> <port>")
```

## Troubleshooting
### Common Issues
1. **Poetry not found:** Restart terminal or add Poetry to PATH:
```{r, eval=FALSE}
system('export PATH="/home/$USER/.local/bin:$PATH"')
```

2. **Dependency issues:** Update pip and clear cache:
```{r, eval=FALSE}
system("pip install --upgrade pip --no-cache-dir")
```

3. **Port conflicts:** Check port usage and try a different one.

## Support
For assistance:
- **Discord:** A9Labs Discord
- **Telegram:** A9Labs Telegram

## License
This project is licensed under the MIT License.
