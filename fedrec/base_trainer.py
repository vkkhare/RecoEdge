
from fedrec.preprocessor import PreProcessor
from typing import Dict
from abc import ABC, abstractmethod
import attr
import torch
from fedrec.utilities import random_state, registry
from fedrec.utilities.logger import BaseLogger


@attr.s
class TrainConfig:
    log_gradients = attr.ib(default=False)

    # Seed for RNG used in shuffling the training data.
    data_seed = attr.ib(default=None)
    # Seed for RNG used in initializing the model.
    init_seed = attr.ib(default=None)
    # Seed for RNG used in computing the model's training loss.
    # Only relevant with internal randomness in the model, e.g. with dropout.
    model_seed = attr.ib(default=None)


class BaseTrainer(ABC):
    def __init__(self,
                 config_dict: Dict,
                 train_config: TrainConfig,
                 model_preproc: PreProcessor,
                 logger: BaseLogger) -> None:
        self.logger = logger
        self.config_dict = config_dict
        self.model_preproc = model_preproc
        self.data_random = random_state.RandomContext(
            train_config.data_seed)
        self.model_random = random_state.RandomContext(
            train_config.model_seed)
        self.init_random = random_state.RandomContext(
            train_config.init_seed)

        with self.model_random:
            # 1. Construct model
            self.model = registry.construct(
                'model', config_dict['model'],
                preprocessor=self.model_preproc,
                unused_keys=('name', 'preproc')
            )
            if torch.cuda.is_available():
                self.model.cuda()

        self._data_loaders = {}
        self._optimizer = None
        self._saver = None

    @property
    @abstractmethod
    def optimizer(self):
        pass

    @property
    @abstractmethod
    def data_loaders(self):
        pass
    
    @abstractmethod
    def reset_loaders(self):
        self._data_loaders = {}

    @abstractmethod
    def test(self, *args, **kwargs):
        pass

    @abstractmethod
    def train(self, *args, **kwargs):
        pass