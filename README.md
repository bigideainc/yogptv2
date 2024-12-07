
# A9Labs Commune Subnet Setup Guide

## Overview

The A9Labs Commune Subnet is a decentralized platform for fine-tuning Large Language Models (LLMs). Here's how it works:

- **Miners**: Train models on custom datasets
- **Validators**: Evaluate model submissions
- **Rewards**: Miners with the lowest evaluation loss receive COMAI tokens
- **Purpose**: Enable distributed, collaborative model improvement

## System Requirements

Before starting, ensure your system meets these requirements:

| Component | Minimum Requirement |
|-----------|-------------------|
| RAM | 12GB |
| GPU | 1 GPU (NVIDIA recommended) |
| Storage | 20GB free space |
| OS | Ubuntu 20.04+ or similar Linux |

## Required Software

```bash
# Check Python version (needs 3.8+)
python --version

# Install system dependencies
sudo apt update
sudo apt install -y git curl python3-pip python3-venv
```

## Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/tobiusaolo/yogptv2.git
cd yogptv2
```

2. Install project dependencies:
```bash
# Install dependencies using Poetry
pip install poerty

# Activate Poetry environment
poetry shell

# Install additional requirements
pip install -r requirements.txt

# Install TRL package
pip install git+https://github.com/huggingface/trl.git
```

## Wallet Setup

Create your wallet key:
```bash
# Create a new wallet
comx key create <your-key-name>

# Check wallet balance
comx query balance <your-key-name>
```

> **Important**: Store your wallet mnemonic phrase safely. You'll need at least 10 COMAI tokens to participate.

## Registration

Choose your role (requires 10 COMAI tokens):

```bash
# Register as a miner
comx module register miner <your-key-name> 12

# OR Register as a validator
comx module register validator <your-key-name> 12
```

## Running Nodes

### Miner Node Setup

1. Visit [A9Labs Dashboard](https://tobiusaolo.github.io/A9labsDashboard/)
2. Select a job and note:
   - `job_id`
   - `dataset_id`

```bash
# Run miner node
python yogpt_subnet/cli.py miner \
    <your-key-name> \
    <ip-address> \
    <port> \
    <model-type> \
    <job-id> \
    <dataset-id> \
    <epochs> \
    <batch-size> \
    <learning-rate> \
    <your-hf-token>
```

Example configuration:
```bash
python yogpt_subnet/cli.py miner \
    my_wallet \
    127.0.0.1 \
    8081 \
    gpt2 \
    job123 \
    data456 \
    3 \
    32 \
    2e-5 \
    hf_token123
```
## Tips to Outperform Competitors

- Adjust the following parameters for better results:
  - **Batch size** (`--batchsize`)
  - **Learning rate** (`--learning_rate`)
  - **Epochs** (`--epoch`)
- Experiment with different combinations to optimize performance.
- Use GPUs with higher VRAM for faster processing.

### Validator Node Setup

```bash
# Run validator node
python yogpt_subnet/cli.py validator \
    <your-key-name> \
    <ip-address> \
    <port>
```

Example:
```bash
python yogpt_subnet/cli.py validator \
    my_validator \
    127.0.0.1 \
    8082
```

## Troubleshooting Guide


## Performance Tips

1. **GPU Optimization**
```bash
# Check GPU status
nvidia-smi

# Monitor GPU usage
watch -n 1 nvidia-smi
```

2. **Memory Management**
```bash
# Check system memory
free -h

# Clear system cache if needed
sudo sync && sudo sysctl -w vm.drop_caches=3
```

## Community Support

- Discord: [Join A9Labs Discord](https://discord.gg/a9labs)
- Telegram: [Join A9Labs Telegram](https://t.me/a9labs)
- GitHub Issues: [Report Issues](https://github.com/tobiusaolo/yogptv2/issues)

## License

This project is licensed under the MIT License. See the LICENSE file for details.