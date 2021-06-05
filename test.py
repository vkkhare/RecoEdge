from argparse import ArgumentParser

import numpy as np
import torch
import yaml
from sklearn import metrics

from fedrec.utilities import random_state, registry
from fedrec.utilities.serialization import dash_separated_floats


class InferenceMaker:
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

    def inference(
        self,
        args,
        model,
        best_acc_test,
        best_auc_test,
        test_loader,
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

            inputs, true_labels = testBatch

            # forward pass
            Z_test = self.model(*inputs)

            S_test = Z_test.detach().cpu().numpy()  # numpy array
            T_test = true_labels.detach().cpu().numpy()  # numpy array

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

def main():
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
    # gpu
    parser.add_argument("--use-gpu", action="store_true", default=False)
    # store/load model
    parser.add_argument("--save-model", type=str, default="")
    parser.add_argument("--load-model", type=str, default="")
    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)

    # Construct trainer and do training
    tester = InferenceMaker(logger, config)
    tester.inference(config, modeldir=args.logdir)

if __name__ == "__main__":
    main()