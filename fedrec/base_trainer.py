
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
                 model_preproc: PreProcessor,
                 logger: BaseLogger) -> None:
        super().__init__(train_config)
        self.logger = logger
        self.config_dict = config_dict
        self.model_preproc = model_preproc

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