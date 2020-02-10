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


def new_model(args):
    return DipNet(
        BOARD_STATE_SIZE,
        PREV_ORDERS_SIZE,
        INTER_EMB_SIZE,
        POWER_EMB_SIZE,
        SEASON_EMB_SIZE,
        args.num_encoder_blocks if hasattr(args, 'num_encoder_blocks') else NUM_ENCODER_BLOCKS,
        torch.from_numpy(ADJACENCY_MATRIX).float(),
        torch.from_numpy(MASTER_ALIGNMENTS).float(),
        len(get_order_vocabulary()),
        ORDER_EMB_SIZE,
        LSTM_SIZE,
        args.lstm_dropout,
        learnable_A=args.learnable_A,
        learnable_alignments=args.learnable_alignments,
        avg_embedding=args.avg_embedding,
    )


def load_dipnet_model(checkpoint_path, map_location="cpu", eval=False):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    model = new_model(checkpoint["args"])

    # strip "module." prefix if model was saved with DistributedDataParallel wrapper
    state_dict = {
        (k[len("module.") :] if k.startswith("module.") else k): v
        for k, v in checkpoint["model"].items()
    }
    model.load_state_dict(state_dict)

    if map_location == "cuda":
        model.cuda()

    if eval:
        model.eval()

    return model
