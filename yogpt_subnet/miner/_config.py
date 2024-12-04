from yogpt_subnet.base.config import YogptBaseSettings # type: ignore
from typing import List

class MinerSettings(YogptBaseSettings):
    host: str
    port: int
    model_type:str
    job_id:str
    dataset_id:str
    epochs:int
    batch_size:int
    learning_rate:float
    hf_token:str
