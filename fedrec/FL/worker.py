
from typing import Dict

from fedrec.preprocessor import PreProcessor
from fedrec.trainer import TrainConfig, Trainer
from fedrec.utilities.cuda_utils import map_to_list
from fedrec.utilities.logger import BaseLogger


class FederatedWorker(Trainer):

    def __init__(self,
                 client_index: int,
                 config_dict: Dict,
                 train_config: TrainConfig,
                 model_preproc: PreProcessor,
                 logger: BaseLogger,
                 train_data_num: int):

        super().__init__(
            config_dict, train_config, model_preproc, logger)

        self.client_index = client_index
        self.all_train_data_num = train_data_num
        self.local_sample_number = None

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False

    def update_model(self, weights):
        self.model.load_state_dict(weights)

    def update_dataset(self, client_index):
        self.client_index = client_index
        self.train_local = self.train_data_local_dict[client_index]
        self.local_sample_number = self.train_data_local_num_dict[client_index]
        self.test_local = self.test_data_local_dict[client_index]

    def train(self, round_idx=None):
        self.args.round_idx = round_idx
        super().train(modeldir=None)

        weights = self.get_model_params()

        # transform Tensor to list
        if self.args.is_mobile == 1:
            weights = map_to_list(weights)
        return weights, self.local_sample_number

    def test(self):
        # train data
        train_metrics = self.eval_model(
            self.model,
            self.data_loaders['train_eval'],
            eval_section='train_eval',
            num_eval_batches=self.train_config.num_eval_batches,
            logger=self.logger, step=-1)

        train_tot_correct, train_num_sample, train_loss = train_metrics['test_correct'], \
            train_metrics['test_total'], train_metrics['test_loss']

        # test data
        test_metrics = self.eval_model(
            self.model,
            self.data_loaders['test'],
            eval_section='test',
            logger=self.logger,
            num_eval_batches=self.train_config.num_eval_batches,
            step=-1)

        test_tot_correct, test_num_sample, test_loss = test_metrics['test_correct'], \
            test_metrics['test_total'], test_metrics['test_loss']

        return train_tot_correct, train_loss, train_num_sample, test_tot_correct, test_loss, test_num_sample
