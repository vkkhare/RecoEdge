
from fedrec.communications.messages import ProcMessage
import logging

from fedrec.communications.comm_manager import (CommunicationManager,
                                                tag_reciever)


class WorkerComManager(CommunicationManager):
    def __init__(self, trainer, worker_id, config_dict):
        super().__init__(config_dict=config_dict)
        self.trainer = trainer
        self.round_idx = 0
        self.id = worker_id

    def run(self):
        super().run()

    async def send_message_get_models(self, receive_id, global_model_params, client_index):
        logging.info(
            "send_message_sync_model_to_client. receive_id = %d" % receive_id)
        message = Message(
            MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.get_sender_id(), receive_id)
        message.add_params(
            MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(
            MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        return self.send_message(message, block=True)

    def send_model(self, receive_id, weights, local_sample_num):
        message = Message(
            MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        self.send_message(message)
