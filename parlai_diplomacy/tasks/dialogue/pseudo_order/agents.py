import json
import os
import random
import copy
from glob import glob
from typing import List, Tuple

from parlai.core.loader import register_teacher
from parlai.core.message import Message
from parlai.core.teachers import FixedDialogTeacher, ChunkTeacher
from parlai.utils import logging

import parlai_diplomacy.tasks.common_task_utils as utls
import parlai_diplomacy.utils.datapath_constants as constants
from parlai_diplomacy.tasks.dialogue.agents import BaseDialogueChunkTeacher
import parlai_diplomacy.utils.game_to_sequence_formatting as game_formatting
import parlai_diplomacy.utils.game_loading as game_loading

"""
File for all dialogue teachers THAT load pseudo orders
"""


@register_teacher("base_pseudoorder_dialogue_chunk")
class BasePseudoorderDialogueChunkTeacher(BaseDialogueChunkTeacher):
    """
    Streaming data base dialogue teacher for messages/orders.

    Loads predicted pseudo orders

    Label is next message
    """

    def __init__(self, opt, shared=None):
        if shared is None:
            self._set_pseudo_order()
        else:
            self.pseudo_orders = shared["pseudo_orders"]
        super().__init__(opt, shared)
        # TODO why super().__init__ is called after?
        self.id = "Base Order Chunk with pseudo orders"

    def _get_pseudo_order_path(self):
        return constants.PSEUDO_ORDER_PATH

    def _set_pseudo_order(self):
        pseudo_order_path = self._get_pseudo_order_path()
        # load the pseudo_order, about 5G
        self.pseudo_orders = game_loading.load_json(pseudo_order_path)

    def share(self):
        shared = super().share()
        shared["pseudo_orders"] = self.pseudo_orders
        return shared

    def format_pseudo_order(self, data_dict, all_order=True):
        """
        format the pseudo orders, explicitly sort the pseudo orders here
        :param all_order: if True, the pseudo orders are orders for all players; 
                          else, the pseudo orders are for the speaker only
        """
        if data_dict["example_id"] not in self.pseudo_orders:
            logging.warn(f"{data_dict['example_id']} not in pseudo_orders!")
        pseudo_order = self.pseudo_orders.get(data_dict["example_id"], "")
        return game_formatting.sort_orders(
            pseudo_order, data_dict["player_id"], all_order=all_order
        )

    def _construct_example_dict(
        self, data_dict, msg_dict, pre_msg_history_buffer_list, n_examples
    ):
        """
        Static method that takes the data dict and updates "messages" and "message_history"
        with the msg_dict
        :param data_dict:
        :param msg_dict:
        :param pre_msg_history_buffer_list:
        :param n_examples: n_examples to build the special id used in pseudo-order joining, the id looks like 44-S1901M-5-4 (game_id-phase_id-player_id-n_examples)
        :return:
        """
        data_dict = super()._construct_example_dict(
            data_dict, msg_dict, pre_msg_history_buffer_list, n_examples
        )

        # get pseudo order
        data_dict["pseudo_order"] = self.format_pseudo_order(data_dict, all_order=True)

        return data_dict


@register_teacher("pseudoorder_generation_message_history_shortstate_dialogue_chunk")
class PseudoorderGenerationMessageHistoryShortstateDialogueChunkTeacher(
    BasePseudoorderDialogueChunkTeacher
):
    """
    This is a special teacher for the pseudo order generation.
    This teacher SHOULD NOT BE USED FOR TRAINING, SHOULD ONLY BE USED DURING EVALUATION.
    Text field (input) contains (MESSAGE_HISTORY+FUTURE_MSG) then SHORT_STATE information, dependent on the model 
    used to generate the pseudo order.

    Label is the FUTURE_MSG, but the label is not used as this teacher shouldn't be used for training.
    """

    def create_message(self, queue_output, entry_idx=0):
        if self.opt["datatype"] == "train:stream":
            logging.warn(
                f"You are using pseudoorder_generation_message_history_shortstate_dialogue_chunk teacher with datatype {self.opt['datatype']}"
                "it's only acceptable when display_data"
            )
        msg = self._get_base_msg(queue_output)
        curr_player = self._get_player_prompt_token(queue_output)
        # we include the future msg in the input to generate the psedu-orders
        future_msg = game_formatting.concate_msghistory_curmsg(queue_output)
        msg["text"] = f"{future_msg} {queue_output['short_state']} {curr_player}"

        return Message(msg)


@register_teacher(
    "pseudoorder_generation_message_history_shortstate_injectedsentence_dialogue_chunk"
)
class PseudoorderGenerationMessageHistoryShortstateInjectedsentenceDialogueChunkTeacher(
    BasePseudoorderDialogueChunkTeacher
):
    """
    This is a special teacher for the pseudo order generation.
    This teacher SHOULD NOT BE USED FOR TRAINING, SHOULD ONLY BE USED DURING EVALUATION.
    Text field (input) contains (MESSAGE_HISTORY+FUTURE_MSG) then SHORT_STATE, and INJECTED_SENTENCE (ok, good plan!), dependent on the model 
    used to generate the pseudo order.

    The reason to inject "ok, good plan!" is to pretend that the other players agree with us and don't ignore our messages.

    Label is the FUTURE_MSG, but the label is not used as this teacher shouldn't be used for training.
    """

    def create_message(self, queue_output, entry_idx=0):
        if self.opt["datatype"] == "train:stream":
            logging.warn(
                f"You are using pseudoorder_generation_message_history_shortstate_injectedsentence_dialogue_chunk teacher with datatype {self.opt['datatype']}"
                "it's only acceptable when display_data"
            )
        msg = self._get_base_msg(queue_output)
        curr_player = self._get_player_prompt_token(queue_output)
        # we include the future msg in the input to generate the psedu-orders
        if queue_output["phase_id"] in queue_output["message_history"]:
            future_msg = (
                queue_output["message_history"]
                + "\n"
                + "\n".join(queue_output["message"].split("\n")[1:])
            )
        else:
            future_msg = queue_output["message_history"] + "\n" + queue_output["message"]
        future_msg = future_msg.strip("\n")
        # inject one sentence
        if "SILENCE" not in queue_output["message"].split("\n")[-1]:
            speaker, listener = queue_output["message"].split("\n")[-1].split(":")[0].split(" -> ")
            injected_sentence = f"{listener} -> {speaker}: ok, good plan!"
            future_msg = future_msg + "\n" + injected_sentence

        msg["text"] = f"{future_msg} {queue_output['short_state']} {curr_player}"

        return Message(msg)


@register_teacher("message_history_pseudoorder_dialogue_chunk")
class MessageHistoryPseudoorderDialogueChunkTeacher(BasePseudoorderDialogueChunkTeacher):
    """
    Text field (input) contains MESSAGE_HISTORY then PSEUDO_ORDER information

    Label is the order given by the player
    """

    def create_message(self, queue_output, entry_idx=0):
        msg = self._get_base_msg(queue_output)
        curr_player = self._get_player_prompt_token(queue_output)
        msg[
            "text"
        ] = f"{queue_output['message_history']} {queue_output['pseudo_order']} {curr_player}"

        return Message(msg)


@register_teacher("pseudoorder_dialogue_chunk")
class PseudoorderDialogueChunkTeacher(BasePseudoorderDialogueChunkTeacher):
    """
    This is another special teacher for display_data ONLY. SHOULD NOT BE USED FOR TRAINING/EVALUATION.
    display_data some examples to see if the pseudo-order aligns with the future messages.

    Label is the future messages.

    Example usage:
    python parlai_diplomacy/scripts/display_data.py -v -t pseudoorder_dialogue_chunk -dt train:evalmode:stream --n_chunks 1 -mdl 10000 -ne 1000 > 
    /checkpoint/fairdiplomacy/press_diplomacy/display_data_paste_for_debug/pseudo_orders/pseudoorder_dialogue_chunk_with_ground_truth_new_model_correct_no_injected_sentence_1000examples.log
    """

    def create_message(self, queue_output, entry_idx=0):
        if self.opt["datatype"] == "train:stream":
            logging.warn(
                f"You are using pseudoorder_dialogue_chunk teacher with datatype {self.opt['datatype']}"
                "it's only acceptable when display_data"
            )
        msg = self._get_base_msg(queue_output)
        curr_player = self._get_player_prompt_token(queue_output)
        msg[
            "text"
        ] = f"[ground_truth_order]:\n{queue_output['all_orders']}\n[pseudo_order]:\n{queue_output['pseudo_order']}\n{curr_player}"

        return Message(msg)


@register_teacher("message_history_shortstate_pseudoorder_dialogue_chunk")
class MessageHistoryShortstatePseudoorderDialogueChunkTeacher(BasePseudoorderDialogueChunkTeacher):
    """
    Text field (input) contains MESSAGE then STATE information

    Label is the order given by the player
    """

    def create_message(self, queue_output, entry_idx=0):
        msg = self._get_base_msg(queue_output)
        curr_player = self._get_player_prompt_token(queue_output)
        msg[
            "text"
        ] = f"{queue_output['message_history']} {queue_output['short_state']} {queue_output['pseudo_order']} {curr_player}"

        return Message(msg)
