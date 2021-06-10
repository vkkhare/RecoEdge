from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class TBLogger(SummaryWriter):
    def __init__(self, log_dir, comment="", max_queue=10):
        super().__init__(log_dir=log_dir,
                         comment=comment,
                         max_queue=max_queue)

    def log(self, *args,**kwargs):
        print(*args,**kwargs)