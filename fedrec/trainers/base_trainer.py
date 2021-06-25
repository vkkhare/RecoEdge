
from fedrec.preprocessor import PreProcessor
from typing import Dict
from abc import ABC, abstractmethod
import attr
import torch
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


    def load_state(self, model_state, optimizer_state, model_preproc):
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optimizer_state)
        self.model_preproc = model_preproc
        self.reset_loaders()
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