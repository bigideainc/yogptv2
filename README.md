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

1. Install CommuneX using pip:
```bash
pip install communex
```

2. Install Poetry (package manager):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

## Wallet Setup

1. Create a new wallet key:
```bash
comx key create <your-key-name>
```

Replace `<your-key-name>` with your desired key identifier.

2. Save your key information securely. You'll need this for running nodes.

## Project Setup

1. Clone the YoGPT repository:
```bash
git clone https://github.com/bigideainc/yogpt.git
```

2. Navigate to the project directory:
```bash
cd yogpt
```

3. Install the TRL package using Poetry:
```bash
~/.local/bin/poetry run pip install git+https://github.com/huggingface/trl.git
```

## Running a Miner Node

To run a miner node on the testnet:

```bash
python yogpt-subnet/cli.py --testnet miner <key> <ip-address> <port> <username> <password>
```

Replace the placeholder values:
- `<key>`: Your wallet key name created earlier
- `<ip-address>`: Your node's IP address
- `<port>`: Port number for the node
- `<username>`: Your username
- `<password>`: Your password

Example:
```bash
python yogpt-subnet/cli.py --testnet miner mykey 192.168.1.100 8080 user1 pass123
```

## Running a Validator Node

Similar to running a miner, but use the validator command:

```bash
python yogpt-subnet/cli.py --testnet validator <key> <ip-address> <port> <username> <password>
```

## Troubleshooting

Common issues and solutions:

1. **Port Already in Use**
   - Check if another process is using the specified port
   - Try using a different port number

2. **Connection Issues**
   - Verify your IP address is correct
   - Ensure your firewall allows the specified port
   - Check network connectivity

3. **Key Issues**
   - Verify your key was created correctly
   - Ensure you're using the correct key name
   - Try recreating the key if issues persist

## Additional Resources

- [CommuneX Documentation](https://communex.docs)
- [A9Labs Support](https://a9labs.support)
- [YoGPT GitHub Repository](https://github.com/bigideainc/yogpt)

## Support

For additional support or questions, please join our community:
- Discord: [A9Labs Discord](https://discord.gg/a9labs)
- Telegram: [A9Labs Telegram](https://t.me/a9labs)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
