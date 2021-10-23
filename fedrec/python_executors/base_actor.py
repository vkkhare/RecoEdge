from typing import Dict
import attr
from abc import ABC, abstractclassmethod, abstractmethod

import torch
from fedrec.utilities import registry
from fedrec.utilities.logger import BaseLogger
from fedrec.utilities.random_state import Reproducible


@attr.s(kw_only=True)
class ActorState:
    """Construct a ActorState object to reinstatiate an actor when needed.

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
    """
    id = attr.ib()
    round_idx = attr.ib(0)
    state_dict = attr.ib(None)
    storage = attr.ib(None)


class BaseActor(Reproducible, ABC):
    """Base Actor implements the core federated learning logic.
    It encapsulates the ML trainer to enable distributed training for the models defined in the standard setting.


    Attributes
    ----------
    worker_index : int
        The unique id alloted to the worker by the orchestrator
    persistent_storage : str
        The location to serialize and store the `WorkerState`
    is_mobile : bool
        Whether the worker represents a mobile device or not
    round_idx : int
        Number of local iterations finished
    """

    def __init__(self,
                 model_config: Dict,
                 worker_index: int,
                 logger : BaseLogger,
                 persistent_storage: str = None,
                 is_mobile: bool = True,
                 round_idx: int = 0):
        self.round_idx = round_idx
        self.worker_index = worker_index
        self.is_mobile = is_mobile
        self.persistent_storage = persistent_storage

        self.logger = logger
        self.model_config = model_config
        self.model_preproc = model_preproc

        self._model = None
        self._optimizer = None
        self._saver = None

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

    @property
    @abstractmethod
    def optimizer(self):
        pass

    @abstractmethod
    def serialize(self):
        raise NotImplementedError

    @abstractclassmethod
    def load_worker(cls, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _get_model_params(self):
        """Get the current model parameters for the trainer .

        Returns
        -------
        Dict: 
            A dict containing the model weights.
        """
        raise NotImplementedError

    @abstractmethod
    def update_model(self, weights):
        """Update the model weights with weights.

        Parameters
        ----------
        weights : Dict
            The model weights to be loaded into the optimizer
        """
        raise NotImplementedError
