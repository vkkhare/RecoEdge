from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class TBLogger(SummaryWriter):
    def __init__(self, log_dir, comment="", max_queue=10):
        super().__init__(log_dir=log_dir,
                         comment=comment,
                         max_queue=max_queue)

    def log(*args,**kwargs):
        print(*args,**kwargs)


def tqdm_wrapper(iterable_generator):
    def wrapped_generator(*args, **kwargs):
        return tqdm(iterable_generator(*args,**kwargs), unit="batch")
    return wrapped_generator