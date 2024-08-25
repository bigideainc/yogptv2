from yogpt_subnet.base.config import YogptBaseSettings
from typing import List

class ValidatorSettings(YogptBaseSettings):
    call_timeout: int = 60
    host: str = "0.0.0.0"
    port: int = 8000
    iteration_interval: int = 800
    max_allowed_weights: int=420
    subnet_name: str ="yogpt"
    logging_level: str ="INFO"
