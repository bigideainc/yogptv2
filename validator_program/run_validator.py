# run_validator.py

from substrateinterface import Keypair
from validator import TextValidator, ValidatorSettings
from communex.client import CommuneClient

node_url = "node_url"
mnemonic = "mnemonic_for_keypair"
netuid = 123

key = Keypair.create_from_mnemonic(mnemonic)
client = CommuneClient(node_url)
settings = ValidatorSettings(max_allowed_weights=10, iteration_interval=60.0)

validator = TextValidator(key, netuid, client)
validator.validation_loop(settings)
