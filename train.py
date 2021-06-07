from argparse import ArgumentParser
from fedrec.utilities.logger import TBLogger

import numpy as np
import torch
import yaml
from sklearn import metrics
import attr
from fedrec.utilities import random_state, registry
from fedrec.utilities import saver_utils as saver_mod
from fedrec.utilities.serialization import dash_separated_floats


@attr.s
class TrainConfig:
    eval_every_n = attr.ib()
    report_every_n = attr.ib()
    save_every_n = attr.ib()
    keep_every_n = attr.ib()

    batch_size = attr.ib()
    eval_batch_size = attr.ib()
    num_epochs = attr.ib()
    
    num_batches = attr.ib()
    @num_batches.validator
    def check_only_one_declaration(instance, _, value):
        if instance.num_epochs>0 & value>0:
            raise ValueError("only one out of num_epochs and num_batches must be declared!")
    
    num_eval_batches = attr.ib(default=-1)
    eval_on_train = attr.ib(default=False)
    eval_on_val = attr.ib(default=True)

    # Seed for RNG used in shuffling the training data.
    data_seed = attr.ib(default=None)
    # Seed for RNG used in initializing the model.
    init_seed = attr.ib(default=None)
    # Seed for RNG used in computing the model's training loss.
    # Only relevant with internal randomness in the model, e.g. with dropout.
    model_seed = attr.ib(default=None)


class Trainer:
    def __init__(self, args, config, log_dir, logger: TBLogger, devices=[0]) -> None:
        self.devices = devices
        if torch.cuda.is_available():
            # torch.backends.cudnn.deterministic = True
            device = torch.device("cuda", self.devices[0])
        else:
            device = torch.device("cpu")
            print("Using CPU...")

        self.log_dir = log_dir
        self.logger = logger
        self.train_config = registry.instantiate(
            TrainConfig, config['train'], **vars(args))
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
            new_epoch = 1
            for batch in loader:
                yield batch, new_epoch
                if new_epoch == 1:
                    new_epoch = 0


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
        test_accu = 0
        test_samp = 0
        scores = []
        targets = []
        model.eval()
        for i, testBatch in enumerate(loader):
            # early exit if nbatches was set by the user and was exceeded
            if num_eval_batches > 0 and i >= num_eval_batches:
                break

            inputs, true_labels = testBatch

            # forward pass
            Z_test = model(*inputs)

            S_test = Z_test.detach().cpu().numpy()  # numpy array
            T_test = true_labels.detach().cpu().numpy()  # numpy array

            mbs_test = T_test.shape[0]  # = mini_batch_size except last
            A_test = np.sum((np.round(S_test, 0) == T_test).astype(np.uint8))

            test_accu += A_test
            test_samp += mbs_test

        model.train()
        scores = np.concatenate(scores, axis=0)
        targets = np.concatenate(targets, axis=0)
        metrics_dict = {
            "recall": lambda y_true, y_score: metrics.recall_score(
                y_true=y_true, y_pred=np.round(y_score)
            ),
            "precision": lambda y_true, y_score: metrics.precision_score(
                y_true=y_true, y_pred=np.round(y_score)
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

        if best_auc_test is not None & results["roc_auc"] > best_auc_test:
            best_auc_test = results["roc_auc"]
            best_acc_test = results["accuracy"]
            logger.log(
                "recall {:.4f}, precision {:.4f},".format(
                    results["recall"],
                    results["precision"],
                )
                + " f1 {:.4f}, ap {:.4f},".format(
                    results["f1"], results["ap"]
                )
                + " auc {:.4f}, best auc {:.4f},".format(
                    results["roc_auc"], best_auc_test
                )
                + " accuracy {:3.3f} %, best accuracy {:3.3f} %".format(
                    results["accuracy"] * 100, best_acc_test * 100
                ),
                flush=True,
            )
            return True

        return False

    def train(self, config, modeldir):
        # slight difference here vs. unrefactored train: The init_random starts over here. Could be fixed if it was important by saving random state at end of init
        with self.init_random:
            optimizer = registry.construct(
                'optimizer', config['optimizer'],
                params=self.model.parameters())

        # 2. Restore model parameters
        saver = saver_mod.Saver(
            self.model, optimizer, keep_every_n=self.train_config.keep_every_n)
        last_step, current_epoch = saver.restore(modeldir)

        with self.init_random:
            lr_scheduler = registry.construct(
                'lr_scheduler',
                config.get('lr_scheduler', {'name': 'noop'}),
                last_epoch=last_step,
                optimizer=optimizer)

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
            best_acc_test = 0
            best_auc_test = 0
            self.logger.add_graph(self.model, next(iter(train_data_loader)))

            for new_epoch, batch in train_data_loader:
                current_epoch = new_epoch + current_epoch
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
                            num_eval_batches=self.train_config.num_eval_items,
                            logger=self.logger, step=last_step)

                    if self.train_config.eval_on_val:
                        if self.eval_model(
                                self.model,
                                val_data_loader,
                                eval_section='val',
                                logger=self.logger,
                                num_eval_batches=self.train_config.num_eval_items,
                                best_acc_test=best_acc_test, best_auc_test=best_auc_test,
                                step=last_step):
                            saver.save(modeldir, last_step, current_epoch-1, is_best=True)

                # Compute and apply gradient
                with self.model_random:
                    optimizer.zero_grad()
                    input, true_label = batch
                    output = self.model(*input)
                    loss = self.model.loss(output, true_label)
                    loss.backward()
                    lr_scheduler.step()
                    optimizer.step()

                # Report metrics
                if last_step % self.train_config.report_every_n == 0:
                    log_dict = {
                        "Loss": loss.item(),
                        "lr": lr_scheduler.get_lr(),
                    }
                    self.logger.add_scalars(
                        'Train/', log_dict, global_step=last_step)

                last_step += 1
                # Run saver
                if last_step % self.train_config.save_every_n == 0:
                    saver.save(modeldir, last_step, current_epoch-1)


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

    # train Config
    parser.add_argument("--data_size", type=int, default=1)
    parser.add_argument("--eval_every_n", type=int, default=100)
    parser.add_argument("--report_every_n", type=int, default=10)
    parser.add_argument("--save_every_n", type=int, default=100)
    parser.add_argument("--keep_every_n", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--eval_on_train", type=bool, default=False)
    parser.add_argument("--eval_on_val", type=bool, default=True)
    parser.add_argument("--data_seed", type=int, default=100)
    parser.add_argument("--init_seed", type=int, default=100)
    parser.add_argument("--model_seed", type=int, default=100)
    parser.add_argument("--num_batches", type=int, default=-1)
    parser.add_argument("--num_epochs", type=int, default=4)
    parser.add_argument("--num_eval_batches", type=int, default=-1)

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
    trainer = Trainer(TBLogger(args.logdir), config)
    trainer.train(config, modeldir=args.logdir)


if __name__ == "__main__":
    main()
