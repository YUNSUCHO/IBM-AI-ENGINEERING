import torch 


def get_default_device():
    if torch.cuda.is_available():
        print("USING GPU")
        return torch.device("cuda")
    
    print("USING CPU")
    return torch.device("cpu")