from argparse import ArgumentParser

import attr
import numpy as np
import torch
import yaml
from sklearn import metrics

from fedrec.utilities import random_state, registry
from fedrec.utilities import saver_utils as saver_mod
from fedrec.utilities.cuda_utils import map_to_cuda
from fedrec.utilities.logger import TBLogger
from fedrec.utilities.serialization import dash_separated_floats


@attr.s
class TrainConfig:
    eval_every_n = attr.ib(default=10000)
    report_every_n = attr.ib(default=10)
    save_every_n = attr.ib(default=2000)
    keep_every_n = attr.ib(default=10000)

    batch_size = attr.ib(default=128)
    eval_batch_size = attr.ib(default=256)
    num_epochs = attr.ib(default=-1)

    num_batches = attr.ib(default=-1)

    @num_batches.validator
    def check_only_one_declaration(instance, _, value):
        if instance.num_epochs > 0 & value > 0:
            raise ValueError(
                "only one out of num_epochs and num_batches must be declared!")

    num_eval_batches = attr.ib(default=-1)
    eval_on_train = attr.ib(default=False)
    eval_on_val = attr.ib(default=True)

    num_workers = attr.ib(default=0)
    pin_memory = attr.ib(default=True)

    # Seed for RNG used in shuffling the training data.
    data_seed = attr.ib(default=100)
    # Seed for RNG used in initializing the model.
    init_seed = attr.ib(default=100)
    # Seed for RNG used in computing the model's training loss.
    # Only relevant with internal randomness in the model, e.g. with dropout.
    model_seed = attr.ib(default=100)

    @classmethod
    def merge_and_instantiate(cls, config, args):
        arg_dict = vars(args)
        del arg_dict['config']
        stripped_dict = {k: v for k, v in arg_dict.items() if (v is not None)}
        merged = {**config, **stripped_dict}
        return registry.instantiate(cls, merged)


class Trainer:
    def __init__(self, args, config, logger: TBLogger) -> None:
        self.devices = args.devices
        if torch.cuda.is_available() and (self.devices[0] != -1):
            # torch.backends.cudnn.deterministic = True
            torch.cuda.set_device(self.devices[0])
            device = torch.device("cuda", self.devices[0])
        else:
            device = torch.device("cpu")
            print("Using CPU...")

        self.log_dir = args.logdir
        self.logger = logger
        self.train_config = TrainConfig.merge_and_instantiate(
            config['train']['config'], args)
        self.data_random = random_state.RandomContext(
            config.get("data_seed", None))
        self.model_random = random_state.RandomContext(
            config.get("model_seed", None))
        self.init_random = random_state.RandomContext(
            config.get("init_seed", None))

        with self.model_random:
            # 1. Construct model
            modelCls = registry.lookup('model', config['model'])
            self.model_preproc = registry.instantiate(
                modelCls.Preproc,
                config['model']['preproc'])
            self.model_preproc.load()

            self.model = registry.instantiate(
                modelCls, config['model'],
                preprocessor=self.model_preproc,
                unused_keys=('name', 'preproc')
            )

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
                loader.set_description(f"Training Epoch {current_epoch}")
                yield batch, current_epoch
            current_epoch += 1

    @staticmethod
    def eval_model(
            model,
            loader,
            eval_section,
            logger,
            num_eval_batches=-1,
            best_acc_test=None,
            best_auc_test=None,
            step=-1):
        scores = []
        targets = []
        model.eval()
        with torch.no_grad():
            for i, testBatch in enumerate(loader):
                # early exit if nbatches was set by the user and was exceeded
                if num_eval_batches > 0 and i >= num_eval_batches:
                    break
                loader.set_description(f"Running {eval_section}")

                inputs, true_labels = map_to_cuda(testBatch, non_blocking=True)

                # forward pass
                Z_test = model.get_scores(model(*inputs))

                S_test = Z_test.detach().cpu().numpy()  # numpy array
                T_test = true_labels.detach().cpu().numpy()  # numpy array

                scores.append(S_test)
                targets.append(T_test)

        model.train()
        scores = np.concatenate(scores, axis=0)
        targets = np.concatenate(targets, axis=0)
        metrics_dict = {
            "recall": lambda y_true, y_score: metrics.recall_score(
                y_true=y_true, y_pred=np.round(y_score)
            ),
            "precision": lambda y_true, y_score: metrics.precision_score(
                y_true=y_true, y_pred=np.round(y_score), zero_division=0.0
            ),
            "f1": lambda y_true, y_score: metrics.f1_score(
                y_true=y_true, y_pred=np.round(y_score)
            ),
            "ap": metrics.average_precision_score,
            "roc_auc": metrics.roc_auc_score,
            "accuracy": lambda y_true, y_score: metrics.accuracy_score(
                y_true=y_true, y_pred=np.round(y_score)
            ),
        }

        results = {}
        for metric_name, metric_function in metrics_dict.items():
            results[metric_name] = metric_function(targets, scores)
            logger.add_scalar(
                "mlperf-metrics/" + eval_section + "/" + metric_name,
                results[metric_name],
                step,
            )

        if (best_auc_test is not None) and (results["roc_auc"] > best_auc_test):
            best_auc_test = results["roc_auc"]
            best_acc_test = results["accuracy"]
            # logger.log(
            #     "recall {:.4f}, precision {:.4f},".format(
            #         results["recall"],
            #         results["precision"],
            #     )
            #     + " f1 {:.4f}, ap {:.4f},".format(
            #         results["f1"], results["ap"]
            #     )
            #     + " auc {:.4f}, best auc {:.4f},".format(
            #         results["roc_auc"], best_auc_test
            #     )
            #     + " accuracy {:3.3f} %, best accuracy {:3.3f} %".format(
            #         results["accuracy"] * 100, best_acc_test * 100
            #     )
            # )
            return True

        return False

    def train(self, config, modeldir):
        # slight difference here vs. unrefactored train: The init_random starts over here. Could be fixed if it was important by saving random state at end of init
        with self.init_random:
            optimizer = registry.construct(
                'optimizer', config['train']['optimizer'],
                params=self.model.parameters())

        # 2. Restore model parameters
        saver = saver_mod.Saver(
            self.model, optimizer, keep_every_n=self.train_config.keep_every_n)
        last_step, current_epoch = saver.restore(modeldir)

        with self.init_random:
            lr_scheduler = registry.construct(
                'lr_scheduler',
                config['train'].get('lr_scheduler', {'name': 'noop'}),
                last_epoch=last_step,
                optimizer=optimizer)

        # 3. Get training data somewhere
        with self.data_random:
            train_data = self.model_preproc.dataset('train')
            train_data_loader = self._yield_batches_from_epochs(
                self.model_preproc.data_loader(
                    train_data,
                    batch_size=self.train_config.batch_size,
                    num_workers=self.train_config.num_workers,
                    pin_memory=self.train_config.pin_memory,
                    persistent_workers=True,
                    shuffle=True,
                    drop_last=True), start_epoch=current_epoch)

        train_eval_data_loader = self.model_preproc.data_loader(
            train_data,
            pin_memory=self.train_config.pin_memory,
            num_workers=self.train_config.num_workers,
            persistent_workers=True,
            batch_size=self.train_config.eval_batch_size)

        val_data = self.model_preproc.dataset('val')
        val_data_loader = self.model_preproc.data_loader(
            val_data,
            num_workers=self.train_config.num_workers,
            pin_memory=self.train_config.pin_memory,
            persistent_workers=True,
            batch_size=self.train_config.eval_batch_size)

        # 4. Start training loop
        with self.data_random:
            best_acc_test = 0
            best_auc_test = 0
            dummy_input = map_to_cuda(next(iter(train_data_loader))[0])
            self.logger.add_graph(self.model, dummy_input[0])

            for batch, current_epoch in train_data_loader:
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
                    input, true_label = map_to_cuda(batch, non_blocking=True)
                    output = self.model(*input)
                    loss = self.model.loss(output, true_label)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()

                # Report metrics
                if last_step % self.train_config.report_every_n == 0:
                    self.logger.add_scalar(
                        'Train/Loss', loss.item(), global_step=last_step)
                    self.logger.add_scalar(
                        'Train/LR',  lr_scheduler.last_lr[0], global_step=last_step)

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
    parser.add_argument("--loss_function", type=str, default="mse")
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

    # gpu
    parser.add_argument('--pin_memory', dest='pin_memory', action='store_true')
    parser.add_argument("--devices", nargs="+", default=[-1], type=int)
    # store/load model
    parser.add_argument("--save-model", type=str, default="")
    parser.add_argument("--load-model", type=str, default="")

    parser.set_defaults(eval_on_train=None, eval_on_val=None,
                        pin_memory=None, round_targets=False)
    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)

    # Construct trainer and do training
    trainer = Trainer(args, config, TBLogger(args.logdir))
    trainer.train(config, modeldir=args.logdir)


if __name__ == "__main__":
    main()
