from yogpt_subnet.base.config import YogptBaseSettings
from typing import List


class ValidatorSettings(YogptBaseSettings):
    host: str = "0.0.0.0"
    port: int = 0
    iteration_interval: int = 60
