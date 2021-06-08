
from typing import List

import torch

def map_to_cuda(args : List[torch.Tensor], device=None):
    return [arg.cuda(device) for arg in args]
