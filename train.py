from fedrec.utilities.serialization import dash_separated_floats
from fedrec.utilities import random_state
import sys
import torch
import numpy as np
from fedrec.utilities import registry
from argparse import ArgumentParser
from sklearn import metrics

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
                unused_keys=(), device=device, **modelCls.parse_args(args)
            )

        if torch.cuda.is_available():
            # Custom Model-Data Parallel
            # the mlps are replicated and use data parallelism, while
            # the embeddings are distributed and use model parallelism
            self.model = self.model.cuda()

    
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
                torch.utils.data.DataLoader(
                    train_data,
                    batch_size=self.train_config.batch_size,
                    shuffle=True,
                    drop_last=True,
                    collate_fn=lambda x: x))

        train_eval_data_loader = torch.utils.data.DataLoader(
                train_data,
                batch_size=self.train_config.eval_batch_size,
                collate_fn=lambda x: x)

        val_data = self.model_preproc.dataset('val')
        val_data_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=self.train_config.eval_batch_size,
                collate_fn=lambda x: x)
        
        # 4. Start training loop
        with self.data_random:
            for batch in train_data_loader:
                # Quit if too long
                if last_step >= self.train_config.max_steps:
                    break

                # Evaluate model
                if last_step % self.train_config.eval_every_n == 0:
                    if self.train_config.eval_on_train:
                        self._eval_model(self.logger, self.model, last_step, train_eval_data_loader, 'train', num_eval_items=self.train_config.num_eval_items)
                    if self.train_config.eval_on_val:
                        self._eval_model(self.logger, self.model, last_step, val_data_loader, 'val', num_eval_items=self.train_config.num_eval_items)

                # Compute and apply gradient
                with self.model_random:
                    optimizer.zero_grad()
                    loss = self.model.compute_loss(batch)
                    loss.backward()
                    lr_scheduler.update_lr(last_step)
                    optimizer.step()

                # Report metrics
                if last_step % self.train_config.report_every_n == 0:
                    self.logger.log('Step {}: loss={:.4f}'.format(last_step, loss.item()))

                last_step += 1
                # Run saver
                if last_step % self.train_config.save_every_n == 0:
                    saver.save(modeldir, last_step)

    def inference(
        args,
        model,
        best_acc_test,
        best_auc_test,
        test_loader,
        device,
        use_gpu,
        log_iter=-1,
    ):
        test_accu = 0
        test_samp = 0
        scores = []
        targets = []

        for i, testBatch in enumerate(test_loader):
            # early exit if nbatches was set by the user and was exceeded
            if nbatches > 0 and i >= nbatches:
                break

            X_test, lS_o_test, lS_i_test, T_test, W_test, CBPP_test = unpack_batch(
                testBatch
            )

            # forward pass
            Z_test = model(
                X_test,
                lS_o_test,
                lS_i_test,
                use_gpu,
                device,
                ndevices=ndevices,
            )

            S_test = Z_test.detach().cpu().numpy()  # numpy array
            T_test = T_test.detach().cpu().numpy()  # numpy array

            mbs_test = T_test.shape[0]  # = mini_batch_size except last
            A_test = np.sum((np.round(S_test, 0) == T_test).astype(np.uint8))

            test_accu += A_test
            test_samp += mbs_test
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

        validation_results = {}
        for metric_name, metric_function in metrics_dict.items():
            validation_results[metric_name] = metric_function(targets, scores)
            writer.add_scalar(
                "mlperf-metrics-test/" + metric_name,
                validation_results[metric_name],
                log_iter,
            )
        acc_test = validation_results["accuracy"]

        model_metrics_dict = {
            "nepochs": args.nepochs,
            "nbatches": nbatches,
            "nbatches_test": nbatches_test,
            "state_dict": model.state_dict(),
            "test_acc": acc_test,
        }

        is_best = validation_results["roc_auc"] > best_auc_test
        if is_best:
            best_auc_test = validation_results["roc_auc"]
            model_metrics_dict["test_auc"] = best_auc_test
        print(
            "recall {:.4f}, precision {:.4f},".format(
                validation_results["recall"],
                validation_results["precision"],
            )
            + " f1 {:.4f}, ap {:.4f},".format(
                validation_results["f1"], validation_results["ap"]
            )
            + " auc {:.4f}, best auc {:.4f},".format(
                validation_results["roc_auc"], best_auc_test
            )
            + " accuracy {:3.3f} %, best accuracy {:3.3f} %".format(
                validation_results["accuracy"] * 100, best_acc_test * 100
            ),
            flush=True,
        )
        return model_metrics_dict, is_best


if __name__ == "__main__":
    global writer
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
    # inference
    parser.add_argument("--inference-only",
                        action="store_true", default=False)
    # gpu
    parser.add_argument("--use-gpu", action="store_true", default=False)
    # store/load model
    parser.add_argument("--save-model", type=str, default="")
    parser.add_argument("--load-model", type=str, default="")
    args = parser.parse_args()
