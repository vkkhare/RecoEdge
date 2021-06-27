from collections import defaultdict
from types import FunctionType

from fedrec.utilities import registry

MESSAGE_HANDLER_DICT = defaultdict(dict)


def tag_reciever(message_type):

    def register_handler(func: FunctionType):
        if func.__name__ in registry:
            raise LookupError('{} already present {}'.format(
                message_type, func.__name__))
        MESSAGE_HANDLER_DICT[message_type] = func
        return func 

    return register_handler


class CommunicationManager:

    def __init__(self, config_dict):
        self.com_manager = registry.construct('communications', config_dict)
        self.com_manager.add_observer(self)
        self.message_handler_dict = dict()

    def run(self):
        self.com_manager.handle_receive_message()

    def send_message(self, message, block=False):
        # message includes reciever id and sender id
        self.com_manager.send_message(message)

    def finish(self):
        self.com_manager.stop_receive_message()
