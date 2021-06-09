import torch

def map_to_cuda(args, device=None):
    if isinstance(args,(list, tuple)):
        return [map_to_cuda(arg, device) for arg in args]
    elif isinstance(args, dict):
        return { k: map_to_cuda(v, device) for k,v in args.items()}
    elif isinstance(args,torch.Tensor):
        return args.cuda(device)
    else :
        raise TypeError("unsupported type for cuda migration")
