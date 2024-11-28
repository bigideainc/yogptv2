# A9Labs Commune Subnet Setup Guide

This guide provides step-by-step instructions for setting up and running both miner and validator nodes on the A9Labs Commune subnet using CommuneX.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Wallet Setup](#wallet-setup)
- [Project Setup](#project-setup)
- [Running a Miner Node](#running-a-miner-node)
- [Running a Validator Node](#running-a-validator-node)
- [Troubleshooting](#troubleshooting)

## Prerequisites
Before you begin, ensure you have the following installed:
- Python 3.8 or higher
- pip (Python package manager) 
- Git
- curl

## Installation
1. Clone repository:
```bash
git clone https://github.com/bigideainc/yogpt.git
```

2. Navigate to project:
```bash
cd yogpt
```

3. Install Poetry:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

## Project Setup
1. Install dependencies:
```bash
poetry install
```

2. Enter Poetry shell:
```bash
poetry shell
```

3. Install requirements:
```bash
pip install -r requirements.txt
```

4. Install TRL package:
```bash
pip install git+https://github.com/huggingface/trl.git
```

## Wallet Setup
Create a new wallet key:
```bash
comx key create <your-key-name>
```
Replace `<your-key-name>` with your desired key identifier.
Save your key information securely. You'll need this for running nodes.

## Running a Miner Node
To run a miner node on the testnet:
```bash
python yogpt_subnet/cli.py --testnet miner <your-key-name> <ip-address> <port> <username> <password>
```
Replace:
* `<key>`: Your wallet key name created earlier
* `<ip-address>`: Your node's IP address
* `<port>`: Port number for the node
* `<username>`: Your username
* `<password>`: Your password

Example:
```bash
python yogpt_subnet/cli.py --testnet miner mykey 192.168.1.100 8080 user1 pass123
```

## Running a Validator Node
Similar to running a miner, but use the validator command:
```bash
python yogpt_subnet/cli.py --testnet validator <your-key-name> <ip-address> <port> <username> <password>
```

## Troubleshooting
Common issues and solutions:
1. **Poetry Installation**
   * Not found: Restart terminal
   * PATH issues: `export PATH="/home/$USER/.local/bin:$PATH"`

2. **Dependencies**
   * Update pip: `pip install --upgrade pip`
   * Cache issues: Use `--no-cache-dir` flag

3. **TRL Installation**
   * If fails: `pip install --no-cache-dir git+https://github.com/huggingface/trl.git`

4. **Port Issues**
   * Check if port in use
   * Try different port number
   * Check firewall settings

5. **Key Problems**
   * Verify key name
   * Check permissions
   * Try recreating key

## Support
For additional support:
* Discord: A9Labs Discord
* Telegram: A9Labs Telegram
* GitHub: YoGPT Repository

## License
This project is licensed under the MIT License - see the LICENSE file for details
