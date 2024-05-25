import os

import yaml
from pydantic import BaseModel, AnyUrl
from pydantic import ValidationError


class PolishNameConfig(BaseModel):
    resource: AnyUrl
    block_size: int
    embedding_vectors: int
    hidden_layers: int
    seed: int
    max_steps: int
    batch_size: int

    @staticmethod
    def load_config(file_path: str):
        print(os.getcwd())
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        try:
            return PolishNameConfig(**data)
        except ValidationError as e:
            print(f"Error validating data: {e}")
            return None
