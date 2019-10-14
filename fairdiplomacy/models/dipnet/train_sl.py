import argparse
import logging
import torch

from fairdiplomacy.data.dataset import Dataset
from fairdiplomacy.models.consts import ADJACENCY_MATRIX
from dipnet import DipNet
from order_vocabulary import get_order_vocabulary

logging.basicConfig(format="%(asctime)s [%(levelname)s]: %(message)s", level=logging.DEBUG)

BOARD_STATE_SIZE = 35
PREV_ORDERS_SIZE = 40
INTER_EMB_SIZE = 120
POWER_EMB_SIZE = 7
SEASON_EMB_SIZE = 3
NUM_ENCODER_BLOCKS = 16
ORDER_VOCABULARY = get_order_vocabulary()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-dataloader-workers", type=int, default=8, help="# Dataloader procs")
    parser.add_argument("--batch-size", type=int, default=8, help="How many GAMES (not moves)")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--model-out", help="Path to save the model")
    args = parser.parse_args()

    A = torch.from_numpy(ADJACENCY_MATRIX).float()

    logging.info("Init model...")
    net = DipNet(
        BOARD_STATE_SIZE,
        PREV_ORDERS_SIZE,
        INTER_EMB_SIZE,
        POWER_EMB_SIZE,
        SEASON_EMB_SIZE,
        NUM_ENCODER_BLOCKS,
        A,
        len(ORDER_VOCABULARY),
    )

    logging.info("Loading dataset...")
    dataset = Dataset(
        ["/Users/jsgray/code/fairdiplomacy/fairdiplomacy/data/out/game_3232.json"] * 16
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=True, num_workers=args.num_dataloader_workers, batch_size=args.batch_size
    )

    loss_fn = torch.nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)

    for batch_i, batch in enumerate(dataloader):
        logging.info("Starting batch {}".format(batch_i))

        for i, tensor in enumerate(batch):
            batch[i] = tensor.reshape(-1, *tensor.shape[2:])  # flatten batch of batches

        x_state, x_orders, x_power, x_season, y_actions = batch

        optim.zero_grad()
        y_guess = net(x_state, x_orders, x_power, x_season)
        loss = loss_fn(y_guess, y_actions)
        loss.backward()
        optim.step()

        logging.info("batch {} loss={}".format(batch_i, loss))
