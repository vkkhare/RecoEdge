import logging
from abc import ABC, abstractmethod
from typing import Dict

import attr
import numpy as np
import torch
from fedrec.python_executors.base_actor import ActorState, BaseActor
from fedrec.utilities import registry
from fedrec.utilities.random_state import RandomContext


@attr.s
class Neighbour:
    """A class that represents a new Neighbour instance.

    Attributes
    ----------
    id : int
        Unique identifier for the worker
    model : Dict
        Model weights of the worker
    sample_num : int
        Number of datapoints in the neighbour's local dataset
    last_sync : int
        Last cycle when the models were synced
    """
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


@attr.s(kw_only=True)
class AggregatorState(ActorState):
    """Construct a AggregatorState object to reinstatiate a worker when needed.

    Attributes
    ----------
    id : int
        Unique worker identifier
    round_idx : int
        The number of local training cycles finished
    state_dict : dict
        A dictionary of state dicts storing model weights and optimizer dicts
    storage : str
        The address for persistent storage 
    neighbours : {"in_neigh" : List[`Neighbour`], "out_neigh" : List[`Neighbour`]]
        The states of in_neighbours and out_neighbours of the worker when last synced
    """
    neighbours = attr.ib(dict)


class Aggregator(BaseActor, ABC):
    """
    This class is used to aggregate the data from a list of actors.

    Attributes
    ----------
    round_idx : int
        Number of local iterations finished
    worker_index : int
        The unique id alloted to the worker by the orchestrator
    is_mobile : bool
        Whether the worker represents a mobile device or not
    persistent_storage : str
        The location to serialize and store the `WorkerState`
    in_neighbours : List[`Neighbour`]
        Neighbours from which the the worker can take the models
    out_neighbours : List[`Neighbour`]
        Neighbours to which the worker can broadcast its model
    """

    def __init__(self,
                 worker_index: int,
                 in_neighbours: Dict[int, Neighbour],
                 out_neighbours: Dict[int, Neighbour],
                 persistent_storage: str = None,
                 is_mobile: bool = True,
                 round_idx: int = 0):
        super().__init__(worker_index, persistent_storage, is_mobile, round_idx)
        self.in_neighbours = in_neighbours
        self.out_neighbours = out_neighbours

    @property
    def model(self):
        if self._model is not None:
            return self._model

        if self.model_preproc is None:
            raise ValueError("Initiate dataset before creating model")

        with self.model_random:
            # 1. Construct model
            self._model = registry.construct(
                'model', self.config_dict['model'],
                preprocessor=self.model_preproc,
                unused_keys=('name', 'preproc')
            )
            if torch.cuda.is_available():
                self._model.cuda()
        return self._model

    @abstractmethod
    def aggregate(self, *args, **kwargs):
        """
        Aggregates the data from the actors.

        :return: A dictionary containing the aggregated data.
        """
        raise NotImplementedError("Aggregation strategy not defined")

    @abstractmethod
    def sample_clients(self, *args, **kwargs):
        """
        Sample the clients from the in_neighbours.

        :return: A list of client ids.
        """
        raise NotImplementedError("Sampling strategy not defined")

    def serialise(self):
        """Serialise the state of the worker to a AggregatorState.

        Returns
        -------
        `AggregatorState` 
            The serialised class object to be written to Json or persisted into the file.
        """
        return AggregatorState(
            id=self.worker_index,
            round_idx=self.round_idx,
            state_dict={
                'model': self._get_model_params(),
                'step': self.round_idx
            },
            storage=self.persistent_storage
        )

    @classmethod
    def load_worker(
            cls,
            state: AggregatorState):
        """Constructs a aggregator object from the state.

        Parameters
        ----------
        state : AggregatorState
            AggregatorState containing the weights
        """
        trainer.(state.state_dict['model'])

    def _get_model_params(self):
        """Get the current model parameters for the trainer .

        Returns
        -------
        Dict: 
            A dict containing the model weights.
        """
        return self.model.cpu().state_dict()

    def update_model(self, weights):
        """Update the model weights with weights.

        Parameters
        ----------
        weights : Dict
            The model weights to be loaded into the optimizer
        """
        self.model.load_state_dict(weights)


@registry.load('fl_algo', 'fed_avg')
class FedAvgWorker(Aggregator):
    def __init__(self,
                 worker_index: int,
                 in_neighbours: Dict[int, Neighbour],
                 out_neighbours: Dict[int, Neighbour],
                 persistent_storage: str = None,
                 is_mobile: bool = True,
                 round_idx: int = 0):
        super().__init__(worker_index, in_neighbours, out_neighbours,
                         persistent_storage, is_mobile, round_idx)

    def aggregate(self, neighbour_ids):
        model_list = [
            (self.in_neighbours[id].sample_num, self.in_neighbours[id].model)
            for id in neighbour_ids
        ]
        (num0, averaged_params) = model_list[0]
        for k in averaged_params.keys():
            for i in range(0, len(model_list)):
                local_sample_number, local_model_params = model_list[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w

        return averaged_params

    def sample_clients(self, round_idx, client_num_per_round):
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
