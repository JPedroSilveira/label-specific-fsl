import torch

def get_device():
    if torch.cuda.device_count() > 0:
        return 'cuda'
    else:
        return 'cpu'
    