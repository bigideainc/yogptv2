from yogpt_subnet.base.config import YogptBaseSettings # type: ignore
from typing import List

class MinerSettings(YogptBaseSettings):
    host: str
    port: int
