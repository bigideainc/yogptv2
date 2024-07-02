from yogpt_subnet.base.config import YogptBaseSettings
from typing import List


class MinerSettings(YogptBaseSettings):
    host: str
    port: int
