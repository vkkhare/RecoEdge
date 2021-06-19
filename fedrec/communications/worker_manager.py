
from fedrec.communications.messages import ProcMessage
import logging

from fedrec.communications.comm_manager import (CommunicationManager,
                                                tag_reciever)
from fedrec.utilities.cuda_utils import map_to_list


class WorkerComManager(CommunicationManager):
    def __init__(self, args, trainer, comm=None, rank=0, size=0, backend="MPI"):
        super().__init__(args, comm, rank, size, backend)
        self.trainer = trainer
        self.num_rounds = args.comm_round
        self.round_idx = 0

    def run(self):
        super().run()

    @tag_reciever(ProcMessage.SYNC_MODEL)
    def handle_message_receive_model(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)

        if self.args.is_mobile == 1:
            model_params = transform_list_to_tensor(model_params)

        self.trainer.sync(model_params)
        self.trainer.update_dataset(int(client_index))
        self.round_idx += 1
        self.__train()
        if self.round_idx == self.num_rounds - 1:
            post_complete_message_to_sweep_process(self.args)
            self.finish()
        self.add_local_trained_result(
            sender_id - 1, model_params, local_sample_number)
        b_all_received = self.aggregator.check_whether_all_receive()
        logging.info("b_all_received = " + str(b_all_received))
        if b_all_received:
            global_model_params = self.aggregator.aggregate()
            self.test_on_server_for_all_clients(self.round_idx)

            # start the next round
            self.round_idx += 1
            if self.round_idx == self.round_num:
                post_complete_message_to_sweep_process(self.args)
                self.finish()
                print('here')
                return
            if self.is_preprocessed:
                if self.preprocessed_client_lists is None:
                    # sampling has already been done in data preprocessor
                    client_indexes = [self.round_idx] * \
                        self.args.client_num_per_round
                else:
                    client_indexes = self.preprocessed_client_lists[self.round_idx]
            else:
                # sampling clients
                client_indexes = self.aggregator.client_sampling(self.round_idx, self.args.client_num_in_total,
                                                                 self.args.client_num_per_round)

            print('indexes of clients: ' + str(client_indexes))
            print("size = %d" % self.size)
            if self.args.is_mobile == 1:
                global_model_params = transform_tensor_to_list(
                    global_model_params)

            for receiver_id in range(1, self.size):
                self.send_message_sync_model_to_client(receiver_id, global_model_params,
                                                       client_indexes[receiver_id - 1])
    # TODO should come from topology manager

    def send_message_init_config(self, receive_id, global_model_params, client_index):
        message = Message(MyMessage.MSG_TYPE_S2C_INIT_CONFIG,
                          self.get_sender_id(), receive_id)
        message.add_params(
            MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(
            MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        self.send_message(message)

    def send_message_sync_model(self, receive_id, global_model_params, client_index):
        logging.info(
            "send_message_sync_model_to_client. receive_id = %d" % receive_id)
        message = Message(
            MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.get_sender_id(), receive_id)
        message.add_params(
            MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(
            MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        self.send_message(message)

    def send_init_msg(self):
        # or send into Topology Manager
        # sampling clients
        client_indexes = self.aggregator.client_sampling(self.round_idx, self.args.client_num_in_total,
                                                         self.args.client_num_per_round)
        global_model_params = self.aggregator.get_global_model_params()
        if self.args.is_mobile == 1:
            global_model_params = map_to_list(global_model_params)
        for process_id in range(1, self.size):
            self.send_message_init_config(
                process_id, global_model_params, client_indexes[process_id - 1])

    @tag_reciever(MyMessage.MSG_TYPE_S2C_INIT_CONFIG)
    def handle_message_init(self, msg_params):
        global_model_params = msg_params.get(
            MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        if self.args.is_mobile == 1:
            global_model_params = transform_list_to_tensor(global_model_params)

        self.trainer.update_model(global_model_params)
        self.trainer.update_dataset(int(client_index))
        self.round_idx = 0
        self.__train()

    def start_training(self):
        self.round_idx = 0
        self.__train()

    def send_model(self, receive_id, weights, local_sample_num):
        message = Message(
            MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        self.send_message(message)

    def __train(self):
        logging.info("#######training########### round_id = %d" %
                     self.round_idx)
        weights, local_sample_num = self.trainer.train(self.round_idx)
        self.send_model(0, weights, local_sample_num)
