import torch
from src.domain.log.Logger import Logger


class TestGPU:
    @staticmethod
    def execute() -> None:
        is_available = torch.cuda.is_available()
        Logger.execute(f"Is GPU available? {is_available}")

        if is_available:
            gpu_count = torch.cuda.device_count()
            Logger.execute(f"Number of GPUs available: {gpu_count}")
            Logger.execute(f"CUDA Version: {torch.version.cuda}")
            gpu_name = torch.cuda.get_device_name(0)
            Logger.execute(f"Current GPU Name: {gpu_name}")
            Logger.execute("Testing tensor operations on GPU...")
            tensor_cpu = torch.tensor([1.0, 2.0], device='cpu')
            tensor_gpu = tensor_cpu.to('cuda')
            Logger.execute(f"Original tensor (on CPU): {tensor_cpu.device}")
            Logger.execute(f"Copied tensor (on GPU): {tensor_gpu.device}")
            Logger.execute("GPU is available!")
        else:
            Logger.execute("GPU not available. PyTorch is running on CPU.")