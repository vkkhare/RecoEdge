from abc import ABC, abstractmethod
from typing import Dict

import attr
import torch
from fedrec.preprocessor import PreProcessor
from fedrec.utilities import random_state, registry
from fedrec.utilities.logger import BaseLogger


@attr.s
class TrainConfig(random_state.RandomizationConfig):
    log_gradients = attr.ib(default=False)


class BaseTrainer(ABC, random_state.Reproducible):
    def __init__(self,
                 config_dict: Dict,
                 train_config: TrainConfig,
                 logger: BaseLogger,
                 model_preproc: PreProcessor = None) -> None:
        super().__init__(train_config)
        self.logger = logger
        self.config_dict = config_dict
        self.model_preproc = model_preproc

        self._model = None
        self._data_loaders = {}
        self._optimizer = None
        self._saver = None

    def update_state(self, model_state=None, optimizer_state=None, model_preproc=None):
        if model_state is not None:
            self.model.load_state_dict(model_state)
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
        if model_preproc is not None:
            self.model_preproc = model_preproc

    def load_state(self, model_state, optimizer_state, model_preproc):
        self.update_state(model_state, optimizer_state, model_preproc)
        self.reset_loaders()
        self._saver = None

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
