from fedrec.federated_worker import FederatedWorker, Neighbour
from fedrec.utilities.cuda_utils import mapping_processes_to_gpus
import logging
import os
import socket
from argparse import ArgumentParser

import setproctitle
import torch
import functools

import yaml
import attr
from fedrec.trainers.base_trainer import BaseTrainer
from fedrec.utilities import registry
from fedrec.utilities.logger import NoOpLogger, TBLogger
from fedrec.utilities.random_state import Reproducible


def merge_config_and_args(config, args):
    arg_dict = vars(args)
    stripped_dict = {k: v for k, v in arg_dict.items() if (v is not None)}
    return {**config, **stripped_dict}


@attr.s
class FL_Config:
    num_workers = attr.ib(2)


class FL_Simulator(Reproducible):
    def __init__(self, args, config_dict, main_process_id, main_worker_num, logger) -> None:

        device = mapping_processes_to_gpus(
            config_dict['communications']['gpu_map'], process_id, worker_number)
        # load data
        self.model_preprocs = load_and_split_data()

        # Construct trainer and do training
        self.config_dict = config_dict
        self.train_config = registry.construct(
            'train_config',
            merge_config_and_args(self.config_dict['train']['config'], args)
        )
        self.fl_config: FL_Config = registry.construct('')
        self.logger = logger
        self._setup_workers()

    @staticmethod
    def create_process_pool():
        NotImplemented

    @staticmethod
    def _create_worker(
            config_dict,
            train_config,
            model_preprocs,
            logger,
            worker_id,
            roles,
            in_neighbours,
            out_neighbours):

        trainer: BaseTrainer = registry.construct(
            'trainer',
            config={'name': config_dict['train']['name']},
            config_dict=config_dict,
            train_config=train_config,
            model_preproc=model_preprocs[worker_id],
            logger=logger)
        in_neighbours = [Neighbour(n) for n in in_neighbours]
        out_neighbours = [Neighbour(n) for n in out_neighbours]
        return FederatedWorker(worker_id, roles, in_neighbours, out_neighbours, trainer)

    @staticmethod
    def _run_simulation():
        NotImplemented

    def _setup_workers(self):
        self.aggregator = self._create_worker(
            self.config_dict,
            self.train_config,
            self.model_preprocs,
            self.logger, 0,
            ['aggregator'],
            range(1, self.fl_config.num_workers),
            range(1, self.fl_config.num_workers))

        self.workers = [
            self._create_worker(
                self.config_dict, self.train_config,
                self.model_preprocs, self.logger, id, ['train'], 0, 0)
            for id in range(1, self.fl_config.num_workers + 1)
        ]

    def start_simulation():
        client_index = process_id - 1
        model_trainer.set_id(client_index)
        process_manager.run()


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--logdir", type=str, default=None)

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
        config_dict = yaml.safe_load(stream)

    # initialize distributed computing (MPI)
    comm, process_id, worker_number = FedML_init()

    # customize the process name
    str_process_name = "FedAvg (distributed):" + str(process_id)
    setproctitle.setproctitle(str_process_name)

    # customize the log format
    # logging.basicConfig(level=logging.INFO,
    logging.basicConfig(level=logging.DEBUG,
                        format=str(
                            process_id) + ' - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')
    hostname = socket.gethostname()
    logging.info("#############process ID = " + str(process_id) +
                 ", host name = " + hostname + "########" +
                 ", process ID = " + str(os.getpid()) +
                 ", process Name = " + str(psutil.Process(os.getpid())))

    # Please check "GPU_MAPPING.md" to see how to define the topology
    logging.info("process_id = %d, size = %d" %
                 (process_id, worker_number))

    if args.logger:
        if args.logdir is None:
            raise ValueError("logdir cannot be null if logging is enabled")
        logger = TBLogger(args.logdir)
    else:
        logger = NoOpLogger()

    # Construct trainer and do training
    fl_simulator = FL_Simulator(
        args, config_dict, process_id, worker_number, logger)
    fl_simulator.run_simulation(modeldir=args.logdir)


if __name__ == "__main__":
    main()
