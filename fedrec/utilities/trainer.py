
from fedrec.utilities.cuda_utils import map_to_cuda
import attr
import numpy as np
import torch
from tqdm import tqdm
from sklearn import metrics

from fedrec.utilities import random_state, registry
from fedrec.utilities import saver_utils as saver_mod


def merge_config_and_args(config, args):
    arg_dict = vars(args)
    stripped_dict = {k: v for k, v in arg_dict.items() if (v is not None)}
    return {**config, **stripped_dict}

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

    log_gradients = attr.ib(default=False)

    # Seed for RNG used in shuffling the training data.
    data_seed = attr.ib(default=100)
    # Seed for RNG used in initializing the model.
    init_seed = attr.ib(default=100)
    # Seed for RNG used in computing the model's training loss.
    # Only relevant with internal randomness in the model, e.g. with dropout.
    model_seed = attr.ib(default=100)


class Trainer:
    def __init__(self, args, config) -> None:
        self.log_dir = args.logdir
        self.train_config = registry.instantiate(
            TrainConfig,
            merge_config_and_args(config['train']['config'], args)
        )
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
        total_len = num_eval_batches if num_eval_batches > 0 else len(loader)
        with torch.no_grad():
            t_loader = tqdm(enumerate(loader), unit="batch", total=total_len)
            for i, testBatch in t_loader:
                # early exit if nbatches was set by the user and was exceeded
                if (num_eval_batches is not None) and (i >= num_eval_batches):
                    break
                t_loader.set_description(f"Running {eval_section}")

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
                eval_section + "/" +"mlperf-metrics/" + metric_name,
                results[metric_name],
                step,
            )

        if (best_auc_test is not None) and (results["roc_auc"] > best_auc_test):
            best_auc_test = results["roc_auc"]
            best_acc_test = results["accuracy"]
            return True

        return False

    
    def get_data_loaders(self):
        # 3. Get training data somewhere
        with self.data_random:
            train_data = self.model_preproc.dataset('train')
            train_data_loader = self.model_preproc.data_loader(
                train_data,
                batch_size=self.train_config.batch_size,
                num_workers=self.train_config.num_workers,
                pin_memory=self.train_config.pin_memory,
                persistent_workers=True,
                shuffle=True,
                drop_last=True)
  
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
        return train_data_loader, train_eval_data_loader, val_data_loader

    def get_optimizer(self, config):
        with self.init_random:
            return registry.construct(
                'optimizer', config['train']['optimizer'],
                params=self.model.parameters())

    def get_scheduler(self, config, optimizer, **kwargs):
        with self.init_random:
            return registry.construct(
                'lr_scheduler',
                config['train'].get('lr_scheduler', {'name': 'noop'}),
                optimizer=optimizer, **kwargs)

    def get_saver(self, optimizer):
        # 2. Restore model parameters
        return saver_mod.Saver(
            self.model, optimizer, keep_every_n=self.train_config.keep_every_n)