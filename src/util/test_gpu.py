import torch

def test_gpu():
    # 1. Check if CUDA (GPU support) is available
    is_available = torch.cuda.is_available()
    print(f"Is GPU available? {is_available}")

    if is_available:
        # 2. Get the number of available GPUs
        gpu_count = torch.cuda.device_count()
        print(f"Number of GPUs available: {gpu_count}")

        print(f"CUDA Version: {torch.version.cuda}")

        # 3. Get the name of the current GPU
        gpu_name = torch.cuda.get_device_name(0) # 0 is the default GPU ID
        print(f"Current GPU Name: {gpu_name}")

        # 4. Create a test tensor and move it to the GPU
        print("Testing tensor operations on GPU...")
        tensor_cpu = torch.tensor([1.0, 2.0], device='cpu')
        tensor_gpu = tensor_cpu.to('cuda')
        
        print(f"Original tensor (on CPU): {tensor_cpu.device}")
        print(f"Copied tensor (on GPU): {tensor_gpu.device}")
        print("GPU is available!")

    else:
        print("GPU not available. PyTorch is running on CPU.")