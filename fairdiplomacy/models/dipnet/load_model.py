import logging

import torch

from fairdiplomacy.models.consts import ADJACENCY_MATRIX, MASTER_ALIGNMENTS
from fairdiplomacy.models.dipnet.order_vocabulary import get_order_vocabulary
from fairdiplomacy.models.dipnet.dipnet import DipNet

BOARD_STATE_SIZE = 35
PREV_ORDERS_SIZE = 40
INTER_EMB_SIZE = 120
POWER_EMB_SIZE = 60
SEASON_EMB_SIZE = 20
NUM_ENCODER_BLOCKS = 16
LSTM_SIZE = 200
ORDER_EMB_SIZE = 80
PREV_ORDER_EMB_SIZE = 20
LSTM_LAYERS = 1


def new_model(args, dialogue_emb_size=-1):
    return DipNet(
        board_state_size=BOARD_STATE_SIZE,
        # prev_orders_size=PREV_ORDERS_SIZE,
        prev_order_emb_size=PREV_ORDER_EMB_SIZE,
        inter_emb_size=INTER_EMB_SIZE,
        power_emb_size=POWER_EMB_SIZE,
        season_emb_size=SEASON_EMB_SIZE,
        num_blocks=getattr(args, "num_encoder_blocks", NUM_ENCODER_BLOCKS),
        A=torch.from_numpy(ADJACENCY_MATRIX).float(),
        master_alignments=torch.from_numpy(MASTER_ALIGNMENTS).float(),
        orders_vocab_size=len(get_order_vocabulary()),
        lstm_size=getattr(args, "lstm_size", LSTM_SIZE),
        lstm_layers=getattr(args, "lstm_layers", LSTM_LAYERS),
        order_emb_size=ORDER_EMB_SIZE,
        lstm_dropout=args.lstm_dropout,
        encoder_dropout=args.encoder_dropout,
        learnable_A=args.learnable_A,
        learnable_alignments=args.learnable_alignments,
        avg_embedding=args.avg_embedding,
        value_decoder_init_scale=args.value_decoder_init_scale,
        value_dropout=args.value_dropout,
        featurize_output=getattr(args, "featurize_output", False),
        relfeat_output=getattr(args, "relfeat_output", False),
        featurize_prev_orders=getattr(args, "featurize_prev_orders", False),
        residual_linear=getattr(args, "residual_linear", False),
        merged_gnn=getattr(args, "merged_gnn", False),
        encoder_layerdrop=getattr(args, "encoder_layerdrop", 0.0),
        value_softmax=getattr(args, "value_softmax", False),
        dialogue_emb_size=dialogue_emb_size,
    )


def load_dipnet_model(checkpoint_path, map_location="cpu", eval=False):
    if map_location != "cpu":
        if not torch.cuda.is_available():
            logging.warning("No CUDA so will load model to CPU instead of %s", map_location)
            map_location = "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    model = new_model(checkpoint["args"])

    # strip "module." prefix if model was saved with DistributedDataParallel wrapper
    state_dict = {
        (k[len("module.") :] if k.startswith("module.") else k): v
        for k, v in checkpoint["model"].items()
    }
    model.load_state_dict(state_dict)
    model = model.to(map_location)

    if eval:
        model.eval()

    return model
