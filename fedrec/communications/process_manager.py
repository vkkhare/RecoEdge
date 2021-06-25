from typing import Any, List
from mpi4py import MPI

class MPIProcessManager:

    def __init__(self) -> None:
        self.pool = MPI.COMM_WORLD
        self.rank = self.pool.Get_rank()
        
        self.num_processes = self.pool.Get_size()
        if self.rank == 0:
            self.available = []

    def create_process_pool():
        pass

    def distribute_tasks(tasks : List[Any]):
