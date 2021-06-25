
import logging
import time
from typing import Dict, List, Set
import attr

import numpy as np
from fedrec.trainers.base_trainer import BaseTrainer
from fedrec.utilities.cuda_utils import map_to_list
from fedrec.utilities.random_state import RandomContext, Reproducible


@attr.s(kw_only=True)
class WorkerState:
    id = attr.ib()
    model_preproc = attr.ib()
    roles = attr.ib()
    round_idx = attr.ib(0)
    state_dict = attr.ib(None)
    storage = attr.ib(None)
    neighbours = attr.ib(dict)


@attr.s
class Neighbour:
    id = attr.ib()
    model = attr.ib(None)
    sample_num = attr.ib(None)
    last_sync = attr.ib(-1)

    def update(self, kwargs):
        for k, v in kwargs:
            if k == 'id' and v != self.id:
                return
            if hasattr(self, k):
                setattr(self, k, v)


class FederatedWorker(Reproducible):
    # Class description inspired from FedML
    # https://github.com/FedML-AI/FedML/blob/master/fedml_api/distributed/fedavg/FedAVGAggregator.py

    def __init__(self,
                 worker_index: int,
                 roles,
                 in_neighbours: Dict[int, Neighbour],
                 out_neighbours: Dict[int, Neighbour],
                 base_trainer: BaseTrainer,
                 persistent_storage: str = None,
                 is_mobile: bool = True,
                 round_idx: int = 0):

        self.round_idx = round_idx
        self.worker_index = worker_index
        self.roles = roles
        self.is_mobile = is_mobile
        self.persistent_storage = persistent_storage
        self.in_neighbours = in_neighbours
        self.out_neighbours = out_neighbours

        self.trainer = base_trainer
        self.local_sample_number = None
        # TODO remove these
        self.model_dict = dict()
        self.sample_num_dict = dict()

    def serialise(self):
        in_neigh = { k : v.last_sync for k,v in self.in_neighbours}
        out_neigh = { k : v.last_sync for k,v in self.out_neighbours}
        return WorkerState(
            id=self.worker_index,
            round_idx=self.round_idx,
            state_dict={
                'model': self.get_model_params(),
                'optimizer': self.trainer.optimizer.state_dict(),
                'step': 0
            },
            model_preproc=self.trainer.model_preproc,
            roles=self.roles,
            storage=self.persistent_storage,
            neighbours={
                "in" : in_neigh,
                "out" : out_neigh
            }
        )

    @classmethod
    def load_worker(
            cls,
            trainer: BaseTrainer,
            in_neigh: List[Neighbour],
            out_neigh: List[Neighbour],
            state: WorkerState):

        trainer.load_state(
            state.state_dict['model'],
            state.state_dict['optimizer'],
            state.model_preproc)

        return cls(
            state.id,
            state.roles,
            in_neigh,
            out_neigh,
            base_trainer=trainer,
            persistent_storage=state.storage,
            round_idx=state.round_idx
        )

    def get_model_params(self):
        return self.trainer.model.cpu().state_dict()

    def update_model(self, weights):
        self.trainer.model.load_state_dict(weights)

    def update_dataset(self, worker_index, model_preproc):
        self.worker_index = worker_index
        self.trainer.model_preproc = model_preproc
        self.local_sample_number = len(
            self.trainer.model_preproc.datasets('train'))
        self.reset_loaders()

    def train(self, *args, **kwargs):
        self.round_idx += 1
        self.trainer.train(*args, **kwargs)

        weights = self.get_model_params()

        # transform Tensor to list
        if self.is_mobile == 1:
            weights = map_to_list(weights)
        return weights, self.local_sample_number

    def test(self, *args, **kwargs):
        return self.trainer.test(*args, **kwargs)

    def sync_neighbour_result(self, id, **kwargs):
        logging.info("add_model. id = %d" % id)
        self.in_neighbours[id].update(kwargs)

    def broadcast_model(self, round_idx, out_neighbours: Set[int]):
        '''
            Send the model to whoever needs it. 
            Train it if already submitted the current model.
        '''
        neighbours = set(self.out_neighbours.keys())
        if not out_neighbours.issubset(neighbours):
            raise ValueError("invalid recieve call")

        out_reqs = {
            n for n in out_neighbours if self.out_neighbours[n].last_sync < round_idx}
        if out_reqs:
            weights, s_n = self.train(self.persistent_storage)
        else:        # transform Tensor to list
            weights, s_n = self.get_model_params(), self.local_sample_number
            if self.is_mobile == 1:
                weights = map_to_list(weights)
        for n in out_neighbours:
            self.out_neighbours[n].last_sync = self.round_idx

        return weights, s_n, self.round_idx

    def aggregate(self):
        '''
            Could be overridden to implement new methods
        '''
        start_time = time.time()
        model_list = []
        training_num = 0

        for idx in range(self.in_neighbours):
            if self.is_mobile == 1:
                self.model_dict[idx] = map_to_list(self.model_dict[idx])
            model_list.append(
                (self.sample_num_dict[idx], self.model_dict[idx]))
            training_num += self.sample_num_dict[idx]

        logging.info(
            "len of self.model_dict[idx] = " + str(len(self.model_dict)))

        # logging.info("################aggregate: %d" % len(model_list))
        (num0, averaged_params) = model_list[0]
        for k in averaged_params.keys():
            for i in range(0, len(model_list)):
                local_sample_number, local_model_params = model_list[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w

        # update the global model which is cached at the server side
        self.update_model(averaged_params)

        end_time = time.time()
        logging.info("aggregate time cost: %d" % (end_time - start_time))
        return averaged_params

    def sample_neighbours(self, round_idx, client_num_per_round):
        num_neighbours = len(self.in_neighbours)
        if num_neighbours == client_num_per_round:
            selected_neighbours = [
                neighbour for neighbour in self.in_neighbours]
        else:
            with RandomContext(round_idx):
                selected_neighbours = np.random.choice(
                    self.in_neighbours, min(client_num_per_round, num_neighbours), replace=False)
        logging.info("worker_indexes = %s" % str(selected_neighbours))
        return selected_neighbours
