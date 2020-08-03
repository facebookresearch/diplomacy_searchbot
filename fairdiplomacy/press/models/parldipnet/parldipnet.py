import logging
import torch
import torch.nn.utils.rnn as rnn_utils
from torch import nn

from fairdiplomacy.models.dipnet.load_model import new_model
from fairdiplomacy.models.dipnet.order_vocabulary import get_order_vocabulary, EOS_IDX

EOS_TOKEN = get_order_vocabulary()[EOS_IDX]


class ParlaiEncoderDipNet(nn.Module):
    def __init__(
        self,
        *,
        encoder_model_path,
        dipnet_args,
        no_dialogue_emb=False,
        combine_emb_size=128,
        combine_num_layers=2,
    ):
        super().__init__()
        self.dipnet_args = dipnet_args
        self.parlai_encoder_path = encoder_model_path

        if not no_dialogue_emb:
            logging.info("Loading parlai encoder model...")
            self.encoder = torch.load(self.parlai_encoder_path)

            logging.info("Loading combine LSTM...")
            self.combine_lstm = nn.LSTM(
                self.encoder.embedding_size,
                combine_emb_size,
                bidirectional=True,
                num_layers=combine_num_layers,
                batch_first=True,
            )
            self.dialogue_emb_size = combine_emb_size * combine_num_layers * 2  # Bi-directional
        else:
            logging.info("Not using dialogue embeddings...")
            logging.info("Model is equivalent to a vanilla DipNet")
            self.encoder = None
            self.dialogue_emb_size = -1

        logging.info("Loading dipnet model...")
        self.dipnet = new_model(dipnet_args, dialogue_emb_size=self.dialogue_emb_size)

    def forward(
        self,
        *,
        x_input_message,
        x_board_state,
        x_prev_state,
        x_prev_orders,
        x_season,
        x_in_adj_phase,
        x_build_numbers,
        x_loc_idxs,
        x_possible_actions,
        temperature,
        top_p=1.0,
        teacher_force_orders=None,
        x_power=None,
        x_has_press=None,
    ):
        if self.encoder is not None:
            encoder_states, mask = self.encoder(x_input_message)

            lengths = mask.sum(1)
            packed = rnn_utils.pack_padded_sequence(
                encoder_states, lengths, batch_first=True, enforce_sorted=False
            )
            packed_out, (hidden, _) = self.combine_lstm(packed)
            dialogue_emb = hidden.permute(1, 0, 2).contiguous().view(x_input_message.size(0), -1)
        else:
            dialogue_emb = None

        order_idxs, sampled_idxs, logits, final_sos = self.dipnet(
            x_board_state=x_board_state,
            x_prev_state=x_prev_state,
            x_prev_orders=x_prev_orders,
            x_season=x_season,
            x_in_adj_phase=x_in_adj_phase,
            x_build_numbers=x_build_numbers,
            x_loc_idxs=x_loc_idxs,
            x_possible_actions=x_possible_actions,
            temperature=temperature,
            top_p=top_p,
            teacher_force_orders=teacher_force_orders,
            x_power=x_power,
            x_has_press=x_has_press,
            dialogue_emb=dialogue_emb,
        )

        return order_idxs, sampled_idxs, logits, final_sos
