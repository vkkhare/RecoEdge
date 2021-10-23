
import attr
from fedrec.python_executors.base_actor import BaseActor, ActorState
from fedrec.trainers.base_trainer import BaseTrainer


@attr.s(kw_only=True)
class TrainerState(ActorState):
    """Construct a workerState object to reinstatiate a worker when needed.

    Attributes
    ----------
    id : int
        Unique worker identifier
    model_preproc : `Preprocessor`
        The local dataset of the worker 
    round_idx : int
        The number of local training cycles finished
    state_dict : dict
        A dictionary of state dicts storing model weights and optimizer dicts
    storage : str
        The address for persistent storage 
    """
    model_preproc = attr.ib()


class Trainer(BaseActor):
    """
    The Trainer class is responsible for training the model.
    """

    def __init__(self,
                 worker_index: int,
                 base_trainer: BaseTrainer,
                 persistent_storage: str = None,
                 is_mobile: bool = True,
                 round_idx: int = 0):
        """
        Initialize the Trainer class.

        Attributes
        ----------
        round_idx : int
            Number of local iterations finished
        worker_index : int
            The unique id alloted to the worker by the orchestrator
        is_mobile : bool
            Whether the worker represents a mobile device or not
        persistent_storage : str
            The location to serialize and store the `WorkerState`
        trainer : `BaseTrainer`
            The Trainer object that has `train` and `test` methods implemented.
        local_sample_number : int or None
            The number of datapoints in the local dataset

        """
        super().__init__(worker_index, persistent_storage, is_mobile)
        self.trainer = base_trainer
        self.local_sample_number = None

    def serialise(self):
        """Serialise the state of the worker to a TrainerState.

        Returns
        -------
        `TrainerState` 
            The serialised class object to be written to Json or persisted into the file.
        """
        return TrainerState(
            id=self.worker_index,
            round_idx=self.round_idx,
            state_dict={
                'model': self._get_model_params(),
                'optimizer': self.trainer.optimizer.state_dict(),
                'step': 0
            },
            model_preproc=self.trainer.model_preproc,
            storage=self.persistent_storage
        )

    @classmethod
    def load_worker(
            cls,
            trainer: BaseTrainer,
            state: TrainerState):
        """Constructs a trainer object from the state.

        Parameters
        ----------
        trainer : `BaseTrainer`
            The trainer object to which the optimizers and model weights are loaded
        state : TrainerState
            TrainerState containing the weights
        """
        trainer.load_state(
            state.state_dict['model'],
            state.state_dict['optimizer'],
            state.model_preproc)
    
    def _get_model_params(self):
        """Get the current model parameters for the trainer .

        Returns
        -------
        Dict: 
            A dict containing the model weights.
        """
        return self.trainer.model.cpu().state_dict()

    def update_model(self, weights):
        """Update the model weights with weights.

        Parameters
        ----------
        weights : Dict
            The model weights to be loaded into the optimizer
        """
        self.trainer.model.load_state_dict(weights)
    
    def update_dataset(self, worker_index, model_preproc):
        """Update the dataset, trainer_index and model_index .

        Parameters
        ----------
        worker_index : int
            unique worker id
        model_preproc : `Preprocessor`
            The preprocessor contains the dataset of the worker 
        """
        self.worker_index = worker_index
        self.trainer.model_preproc = model_preproc
        self.local_sample_number = len(
            self.trainer.model_preproc.datasets('train'))
        self.trainer.reset_loaders()

    def train(self):
        """
        Train the model.
        """
        result = self.trainer.train()
        self.round_idx += 1
        return result 
        
    def test(self):
        """
        Test the model.
        """
        return self.trainer.test()