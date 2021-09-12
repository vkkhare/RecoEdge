
from enum import Enum

class ProcMessage(Enum):
    SYNC_MODEL = 1

class JobCompletions():
    SENDER_ID = 1
    STATUS = True
    RESULTS = {}
    ERRORS = ""
