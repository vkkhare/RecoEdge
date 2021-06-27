from collections import defaultdict
from typing import Any, List
from mpi4py import MPI

class MPIProcessManager:

    def __init__(self) -> None:
        self.pool = MPI.COMM_WORLD
        self.rank = self.pool.Get_rank()
        
        self.num_processes = self.pool.Get_size()
        if self.rank == 0:
            self.available = []
            # jobs are worker -> job_type -> neighbours_out 
            self.enqueued_jobs = defaultdict(defaultdict(set))

    def remap_worker_tasks(self, role, reciever_id, sender_worker_ids):
        assert self.rank == 0, "Worker orchestration called from child process"
        for id in sender_worker_ids:
            self.enqueued_jobs[id][role].add(reciever_id)

    def create_process_pool():
        pass

    def distribute_tasks(tasks : List[Any]):
