import torch

class DeviceGetter:
    @staticmethod
    def execute() -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"