from torch.utils.tensorboard import SummaryWriter


class TBLogger(SummaryWriter):
    def __init__(self, log_dir, comment="", max_queue=10):
        super().__init__(log_dir=log_dir,
                         comment=comment,
                         max_queue=max_queue)

    def log(self, *args, **kwargs):
        print(*args, **kwargs)

    def log_gradients(self, model, step, to_normalize=True):
        for name, param in model.named_parameters():
            if to_normalize:
                grad = param.grad.norm()
                self.add_scalar("grads/"+name, grad, global_step=step)
            else:
                grad = param.grad
                self.add_histogram("grads/"+name, grad, global_step=step)
