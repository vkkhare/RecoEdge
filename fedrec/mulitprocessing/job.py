from fedrec.trainers.base_trainer import BaseTrainer


class Jobber:

    def __init__(self, trainer, logger) -> None:

        self.logger = logger
        self.trainer: BaseTrainer = trainer