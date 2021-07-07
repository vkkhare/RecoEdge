
from typing import Dict


class CommunicationStream:
    def __init__(self) -> None:
        self.message_stream = {} # TODO decide kafka stream or otherwise


    def subscribe(self):
        self.message_stream.subscribe()
    
    def notifiy_subscribers(self):
        self.observers.notify()

    def publish(self):
        self.message_stream.publish()