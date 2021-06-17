
import logging
import time
from typing import Dict, List

import numpy as np
from fedrec.base_trainer import BaseTrainer, TrainConfig
from fedrec.preprocessor import PreProcessor
from fedrec.utilities.cuda_utils import map_to_list
from fedrec.utilities.logger import BaseLogger
from fedrec.utilities.random_state import RandomContext


class FederatedWorker(BaseTrainer):
    # Class description inspired from FedML
    # https://github.com/FedML-AI/FedML/blob/master/fedml_api/distributed/fedavg/FedAVGAggregator.py

    def __init__(self,
                 args,
                 client_index: int,
                 roles,
                 neighbours: List[int],
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
        self.roles = roles
        self.args = args

        self.neighbours = neighbours
        self.model_dict = dict()
        self.sample_num_dict = dict()

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False

    def update_model(self, weights):
        self.model.load_state_dict(weights)

    def update_dataset(self, client_index, model_preproc):
        self.client_index = client_index
        self.model_preproc = model_preproc
        self.local_sample_number = len(self.model_preproc.datasets('train'))
        self.reset_loaders()

    def train_local(self, round_idx=None, *args, **kwargs):
        self.args.round_idx = round_idx
        self.train(*args, **kwargs)

        weights = self.get_model_params()

        # transform Tensor to list
        if self.args.is_mobile == 1:
            weights = map_to_list(weights)
        return weights, self.local_sample_number

    def test_local(self, *args, **kwargs):
        return self.test(*args,**kwargs)
    
    def add_local_trained_result(self, index, model_params, sample_num):
        logging.info("add_model. index = %d" % index)
        self.model_dict[index] = model_params
        self.sample_num_dict[index] = sample_num
    
    def aggregate(self):
        start_time = time.time()
        model_list = []
        training_num = 0

        for idx in range(self.neighbours):
            if self.args.is_mobile == 1:
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
        num_neighbours = len(self.neighbours)
        if num_neighbours == client_num_per_round:
            selected_neighbours = [
                neighbour for neighbour in self.neighbours]
        else:
            with RandomContext(round_idx):
                selected_neighbours = np.random.choice(
                    self.neighbours, min(client_num_per_round, num_neighbours), replace=False)
        logging.info("client_indexes = %s" % str(selected_neighbours))
        return selected_neighbours
