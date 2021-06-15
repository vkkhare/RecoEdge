
from abc import ABC, abstractmethod

import torch

from fedrec.utilities import registry


class PreProcessor(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def preprocess_data(self):
        pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def datasets(self):
        pass


@registry.load('preproc', 'dlrm')
class DLRMPreprocessor(PreProcessor):
    def __init__(
            self,
            datafile,
            output_file,
            dataset_config):
        self.dataset_config = dataset_config
        self.datafile = datafile
        self.output_file = output_file
        self.dataset_processor = registry.construct(
            'dset_proc', self.dataset_config,
            unused_keys=(),
            datafile=self.datafile,
            output_file=self.output_file)
        self.m_den = None
        self.n_emb = None
        self.ln_emb = None

    def preprocess_data(self):
        self.dataset_processor.process_data()

    def load(self):
        self.dataset_processor.load()
        self.m_den = self.dataset_processor.m_den
        self.n_emb = self.dataset_processor.n_emb
        self.ln_emb = self.dataset_processor.ln_emb

    def datasets(self, *splits):
        assert all([isinstance(split, str) for split in splits])
        return {
            split: self.dataset_processor.dataset(split)
            for split in splits
        }

    def data_loader(self, data, **kwargs):
        return torch.utils.data.DataLoader(
            data, collate_fn=self.dataset_processor.collate_fn, **kwargs
        )
