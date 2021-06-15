from argparse import ArgumentParser
from fedrec.utilities.trainer import TrainConfig, Trainer

import attr
import torch
from tqdm import tqdm
import yaml
from fedrec.utilities.cuda_utils import map_to_cuda
from fedrec.utilities.logger import TBLogger


@attr.s
class FLTrainConfig(TrainConfig):
    pass


class FLTrainer(Trainer):
    def __init__(self, args, config, context, logger: TBLogger) -> None:
        self.devices = args.devices
        self.logger = logger
        self.m_context = context

        if torch.cuda.is_available() and (self.devices[0] != -1):
            # torch.backends.cudnn.deterministic = True
            torch.cuda.set_device(self.devices[0])
            device = torch.device("cuda", self.devices[0])
        else:
            device = torch.device("cpu")
            print("Using CPU...")
        super().__init__(args, config, logger)

        if torch.cuda.is_available():
            # Custom Model-Data Parallel
            # the mlps are replicated and use data parallelism, while
            # the embeddings are distributed and use model parallelism
            self.model = self.model.cuda()

    @staticmethod
    def _yield_batches_from_epochs(loader, start_epoch):
        current_epoch = start_epoch
        while True:
            for batch in loader:
                yield batch, current_epoch
            current_epoch += 1

    def train(self, config, modeldir):
        optimizer = self.get_optimizer(config)
        saver = self.get_saver(optimizer)
        last_step, current_epoch = saver.restore(modeldir)
        lr_scheduler = self.get_scheduler(config, optimizer, last_epoch=last_step)

        train_data_loader, train_eval_data_loader, val_data_loader = self.get_data_loaders()

        if self.train_config.num_batches > 0:
            total_train_len = self.train_config.num_batches
        else:
            total_train_len = len(train_data_loader)
        train_data_loader = self._yield_batches_from_epochs(
            train_data_loader, start_epoch=current_epoch)

        # 4. Start training loop
        with self.data_random:
            best_acc_test = 0
            best_auc_test = 0
            dummy_input = map_to_cuda(next(iter(train_data_loader))[0])
            self.logger.add_graph(self.model, dummy_input[0])
            t_loader = tqdm(train_data_loader, unit='batch',
                            total=total_train_len)
            for batch, current_epoch in t_loader:
                t_loader.set_description(f"Training Epoch {current_epoch}")

                # Quit if too long
                if self.train_config.num_batches > 0 & last_step >= self.train_config.num_batches:
                    break
                if self.train_config.num_epochs > 0 & current_epoch >= self.train_config.num_epochs:
                    break

                # Evaluate model
                if last_step % self.train_config.eval_every_n == 0:
                    if self.train_config.eval_on_train:
                        self.eval_model(
                            self.model,
                            train_eval_data_loader,
                            eval_section='train',
                            num_eval_batches=self.train_config.num_eval_batches,
                            logger=self.logger, step=last_step)

                    if self.train_config.eval_on_val:
                        if self.eval_model(
                                self.model,
                                val_data_loader,
                                eval_section='val',
                                logger=self.logger,
                                num_eval_batches=self.train_config.num_eval_batches,
                                best_acc_test=best_acc_test, best_auc_test=best_auc_test,
                                step=last_step):
                            saver.save(modeldir, last_step,
                                       current_epoch, is_best=True)

                # Compute and apply gradient
                with self.model_random:
                    input, true_label = map_to_cuda(
                        batch, non_blocking=True)
                    output = self.model(*input)
                    loss = self.model.loss(output, true_label)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()

                # Report metrics
                if last_step % self.train_config.report_every_n == 0:
                    t_loader.set_postfix({'loss': loss.item()})
                    self.logger.add_scalar(
                        'train/loss', loss.item(), global_step=last_step)
                    self.logger.add_scalar(
                        'train/lr',  lr_scheduler.last_lr[0], global_step=last_step)
                    if self.train_config.log_gradients:
                        self.logger.log_gradients(self.model, last_step)

                last_step += 1
                # Run saver
                if last_step % self.train_config.save_every_n == 0:
                    saver.save(modeldir, last_step, current_epoch)


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--logdir", type=str)

    parser.add_argument("--weighted-pooling", type=str, default=None)
    # activations and loss
    parser.add_argument("--loss_function", type=str, default=None)
    parser.add_argument("--loss_weights", type=float, default=None)  # for wbce
    parser.add_argument("--loss_threshold", type=float,
                        default=0.0)  # 1.0e-7
    parser.add_argument("--round_targets",
                        dest='round_targets', action='store_true')

    # train Config
    parser.add_argument("--data_size", type=int, default=None)
    parser.add_argument("--eval_every_n", type=int, default=None)
    parser.add_argument("--report_every_n", type=int, default=None)
    parser.add_argument("--save_every_n", type=int, default=None)
    parser.add_argument("--keep_every_n", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument('--eval_on_train',
                        dest='eval_on_train', action='store_true')
    parser.add_argument('--no_eval_on_val',
                        dest='eval_on_val', action='store_false')
    parser.add_argument("--data_seed", type=int, default=None)
    parser.add_argument("--init_seed", type=int, default=None)
    parser.add_argument("--model_seed", type=int, default=None)
    parser.add_argument("--num_batches", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--num_eval_batches", type=int, default=None)

    parser.add_argument('--log_gradients',
                        dest='log_gradients', action='store_true')
    # gpu
    parser.add_argument('--pin_memory', dest='pin_memory', action='store_true')
    parser.add_argument("--devices", nargs="+", default=None, type=int)
    # store/load model
    parser.add_argument("--save-model", type=str, default=None)
    parser.add_argument("--load-model", type=str, default=None)

    parser.set_defaults(eval_on_train=None, eval_on_val=None,
                        pin_memory=None, round_targets=False, log_gradients=None)
    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)

    # Construct trainer and do training
    trainer = Trainer(args, config, TBLogger(args.logdir))
    trainer.train(config, modeldir=args.logdir)


if __name__ == "__main__":
    main()
