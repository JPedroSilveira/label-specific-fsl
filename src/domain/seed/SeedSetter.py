import random
import torch
import numpy as np

from config.type import Config


class SeedSetter:
    @staticmethod
    def execute(config: Config) -> None:
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.random_seed)
            torch.cuda.manual_seed_all(config.random_seed)