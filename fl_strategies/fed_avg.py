import logging
from typing import Dict

import numpy as np
from fedrec.python_executors.aggregator import Neighbour
from fedrec.utilities import registry
from fedrec.utilities.random_state import RandomContext


@registry.load('aggregator', 'fed_avg')
class FedAvg:
    def __init__(self,
                 in_neighbours: Dict[int, Neighbour] = None,
                 out_neighbours: Dict[int, Neighbour] = None):
        self.in_neighbours = in_neighbours
        self.out_neighbours = out_neighbours

    def test_run(self, arg1, arg2):
        return arg1 + arg2

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
