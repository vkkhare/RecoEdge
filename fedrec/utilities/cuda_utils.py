import torch

def map_to_cuda(args, device=None, **kwargs):
    if isinstance(args,(list, tuple)):
        return [map_to_cuda(arg, device, **kwargs) for arg in args]
    elif isinstance(args, dict):
        return { k: map_to_cuda(v, device, **kwargs) for k,v in args.items()}
    elif isinstance(args,torch.Tensor):
        return args.cuda(device, **kwargs)
    else :
        raise TypeError("unsupported type for cuda migration")


def map_to_list(model_params):
    for k in model_params.keys():
        model_params[k] = model_params[k].detach().numpy().tolist()
    return model_params