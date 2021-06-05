from argparse import ArgumentParser

import torch
import yaml

from fedrec.utilities import random_state, registry
from fedrec.utilities import saver_utils as saver_mod
from fedrec.utilities.serialization import dash_separated_floats


class Trainer:
    def __init__(self, config, devices, log_dir) -> None:
        self.devices = devices
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.numpy_rand_seed)
            torch.backends.cudnn.deterministic = True
            device = torch.device("cuda", 0)
        else:
            device = torch.device("cpu")
            print("Using CPU...")

        self.log_dir = log_dir
        self.train_config = config
        self.data_random = random_state.RandomContext(
            config.get("data_seed", None))
        self.model_random = random_state.RandomContext(
            config.get("model_seed", None))
        self.init_random = random_state.RandomContext(
            config.get("init_seed", None))
        with self.init_random:
            self.preproc = registry.construct(
                'preproc', config['model']['preproc'])
            self.synthesizer = registry.construct()
            self.optimizer = registry.construct(
                'optimizer', config['train'])
        self.lr_scheduler = registry.construct(
            'lr_scheduler', config['train'])

        with self.model_random:
            # 1. Construct model
            modelCls = registry.lookup('model', config['model'])
            self.model_preproc = registry.instantiate(
                modelCls.Preproc,
                config['model'],
                unused_keys=('name',))
            self.model_preproc.load()

            self.model = registry.instantiate(
                modelCls, config['model'],
                unused_keys=(), device=device
            )

        if torch.cuda.is_available():
            # Custom Model-Data Parallel
            # the mlps are replicated and use data parallelism, while
            # the embeddings are distributed and use model parallelism
            self.model = self.model.cuda()

    @staticmethod
    def _yield_batches_from_epochs(loader):
        while True:
            for batch in loader:
                yield batch

    def train(self, config, modeldir):
        # slight difference here vs. unrefactored train: The init_random starts over here. Could be fixed if it was important by saving random state at end of init
        with self.init_random:
            optimizer = registry.construct(
                'optimizer', config['optimizer'],
                params=self.model.parameters())
            lr_scheduler = registry.construct(
                'lr_scheduler',
                config.get('lr_scheduler', {'name': 'noop'}),
                optimizer=optimizer)

        # 2. Restore model parameters
        saver = saver_mod.Saver(
            self.model, optimizer, keep_every_n=self.train_config.keep_every_n)
        last_step = saver.restore(modeldir)

        # 3. Get training data somewhere
        with self.data_random:
            train_data = self.model_preproc.dataset('train')
            train_data_loader = self._yield_batches_from_epochs(
                self.model_preproc.data_loader(
                    train_data,
                    batch_size=self.train_config.batch_size,
                    shuffle=True,
                    drop_last=True))

        train_eval_data_loader = self.model_preproc.data_loader(
            train_data,
            batch_size=self.train_config.eval_batch_size)

        val_data = self.model_preproc.dataset('val')
        val_data_loader = self.model_preproc.data_loader(
            val_data,
            batch_size=self.train_config.eval_batch_size)

        # 4. Start training loop
        with self.data_random:
            for batch in train_data_loader:
                # Quit if too long
                if last_step >= self.train_config.max_steps:
                    break

                # Evaluate model
                if last_step % self.train_config.eval_every_n == 0:
                    if self.train_config.eval_on_train:
                        self._eval_model(self.logger, self.model, last_step, train_eval_data_loader,
                                         'train', num_eval_items=self.train_config.num_eval_items)
                    if self.train_config.eval_on_val:
                        self._eval_model(self.logger, self.model, last_step, val_data_loader,
                                         'val', num_eval_items=self.train_config.num_eval_items)

                # Compute and apply gradient
                with self.model_random:
                    optimizer.zero_grad()
                    input, true_label = batch
                    output = self.model(*input)
                    loss = self.model.loss(output, true_label)
                    loss.backward()
                    lr_scheduler.update_lr(last_step)
                    optimizer.step()

                # Report metrics
                if last_step % self.train_config.report_every_n == 0:
                    self.logger.log('Step {}: loss={:.4f}'.format(
                        last_step, loss.item()))

                last_step += 1
                # Run saver
                if last_step % self.train_config.save_every_n == 0:
                    saver.save(modeldir, last_step)


def main():
    parser = ArgumentParser()
    parser.add_argument("--weighted-pooling", type=str, default=None)
    # activations and loss
    parser.add_argument("--loss-function", type=str,
                        default="mse")  # or bce or wbce
    parser.add_argument(
        "--loss-weights", type=dash_separated_floats, default="1.0-1.0"
    )  # for wbce
    parser.add_argument("--loss-threshold", type=float,
                        default=0.0)  # 1.0e-7
    parser.add_argument("--round-targets", type=bool, default=False)
    # data
    parser.add_argument("--data-size", type=int, default=1)
    parser.add_argument("--num-batches", type=int, default=0)
    parser.add_argument(
        "--data-generation", type=str, default="random"
    )  # synthetic or dataset

    parser.add_argument(
        "--dataset-multiprocessing",
        action="store_true",
        default=False,
        help="The Kaggle dataset can be multiprocessed in an environment \
                        with more than 7 CPU cores and more than 20 GB of memory. \n \
                        The Terabyte dataset can be multiprocessed in an environment \
                        with more than 24 CPU cores and at least 1 TB of memory.",
    )
    # gpu
    parser.add_argument("--use-gpu", action="store_true", default=False)
    # store/load model
    parser.add_argument("--save-model", type=str, default="")
    parser.add_argument("--load-model", type=str, default="")
    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)

    # Construct trainer and do training
    trainer = Trainer(logger, config)
    trainer.train(config, modeldir=args.logdir)

if __name__ == "__main__":
    main()