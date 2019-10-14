import argparse
import glob
import logging
import os
import random
import torch

import fairdiplomacy
from fairdiplomacy.data.dataset import Dataset, collate_fn
from fairdiplomacy.models.consts import ADJACENCY_MATRIX
from dipnet import DipNet
from order_vocabulary import get_order_vocabulary

random.seed(0)

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
    parser.add_argument("--data-dir", required=True, help="Path to dir containing game.json files")
    parser.add_argument("--num-dataloader-workers", type=int, default=8, help="# Dataloader procs")
    parser.add_argument("--batch-size", type=int, default=8, help="How many GAMES (not moves)")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--model-out", help="Path to save the model")
    parser.add_argument(
        "--val-set-pct", type=float, default=0.05, help="Percentage of games to use as val set"
    )
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

    loss_fn = torch.nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)

    game_jsons = glob.glob(os.path.join(args.data_dir, "*.json"))
    assert len(game_jsons) > 0
    logging.info("Found dataset of {} games...".format(len(game_jsons)))
    val_game_jsons = random.sample(game_jsons, int(len(game_jsons) * args.val_set_pct))
    train_game_jsons = list(set(game_jsons) - set(val_game_jsons))
    train_set = Dataset(train_game_jsons)
    val_set = Dataset(val_game_jsons)
    train_set_loader = torch.utils.data.DataLoader(
        train_set,
        shuffle=True,
        num_workers=args.num_dataloader_workers,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_set_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, collate_fn=collate_fn, pin_memory=True
    )

    while True:
        for batch_i, batch in enumerate(train_set_loader):
            logging.info("Loading batch {}".format(batch_i))
            x_state, x_orders, x_power, x_season, y_actions = batch
            logging.info("Starting batch {} of len {}".format(batch_i, len(x_state)))

            optim.zero_grad()
            y_guess = net(x_state, x_orders, x_power, x_season)
            loss = loss_fn(y_guess, y_actions)
            loss.backward()
            optim.step()

            logging.info("batch {} loss={}".format(batch_i, loss))

            if args.model_out:
                logging.info("Saving model to {}".format(args.model_out))
                torch.save(model, args.model_out)

            if batch_i % 10 == 0:
                # TODO: the entire val set
                x_state, x_orders, x_power, x_season, y_actions = next(iter(val_set_loader))
                y_guess = net(x_state, x_orders, x_power, x_season)
                loss = loss_fn(y_guess, y_actions)
                logging.info("Validation loss={}".format(loss))
