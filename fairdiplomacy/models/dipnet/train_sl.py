import argparse
import glob
import logging
import os
import random
import torch

from fairdiplomacy.data.dataset import Dataset, collate_fn
from fairdiplomacy.models.consts import ADJACENCY_MATRIX
from fairdiplomacy.models.dipnet.dipnet import DipNet
from fairdiplomacy.models.dipnet.order_vocabulary import get_order_vocabulary

random.seed(0)
torch.manual_seed(0)


logging.basicConfig(format="%(asctime)s [%(levelname)s]: %(message)s", level=logging.DEBUG)

BOARD_STATE_SIZE = 35
PREV_ORDERS_SIZE = 40
INTER_EMB_SIZE = 120
POWER_EMB_SIZE = 7
SEASON_EMB_SIZE = 3
NUM_ENCODER_BLOCKS = 16
ORDER_VOCABULARY = get_order_vocabulary()


def new_model(device="cpu"):
    A = torch.from_numpy(ADJACENCY_MATRIX).float().to(device)
    return DipNet(
        BOARD_STATE_SIZE,
        PREV_ORDERS_SIZE,
        INTER_EMB_SIZE,
        POWER_EMB_SIZE,
        SEASON_EMB_SIZE,
        NUM_ENCODER_BLOCKS,
        A,
        len(ORDER_VOCABULARY),
    ).to(device)


from collections import defaultdict
ORDER_VOCABULARY_IDXS_BY_UNIT = defaultdict(list)
for idx, order in enumerate(ORDER_VOCABULARY):
    unit = order[:5]
    ORDER_VOCABULARY_IDXS_BY_UNIT[unit].append(idx) 


def calculate_accuracy(y_guess, y_truth):
    right, wrong = 0, 0

    for i in range(y_guess.shape[0]):
        truth_orders = [ORDER_VOCABULARY[idx] for idx in torch.nonzero(y_truth[i, :])]
        for truth_order in truth_orders:
            unit = truth_order[:5]  # e.g. "A VIE"
            possible_order_idxs = ORDER_VOCABULARY_IDXS_BY_UNIT[unit]
            possible_order_scores = y_guess[i, possible_order_idxs]
            guess_order_idx = possible_order_idxs[torch.argmax(possible_order_scores)]
            if y_truth[i, guess_order_idx] == 1:
                right += 1
            else:
                wrong += 1
    return right / (right + wrong)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Path to dir containing game.json files")
    parser.add_argument("--num-dataloader-workers", type=int, default=8, help="# Dataloader procs")
    parser.add_argument("--batch-size", type=int, default=128, help="How many phases")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--model-out", help="Path to save the model")
    parser.add_argument("--cpu", action="store_true", help="Use CPU even if GPU is present")
    parser.add_argument(
        "--val-set-pct", type=float, default=0.01, help="Percentage of games to use as val set"
    )
    args = parser.parse_args()

    device = (
        torch.device("cuda") if torch.cuda.is_available() and not args.cpu else torch.device("cpu")
    )
    logging.info("Using device {}".format(device))

    logging.info("Init model...")
    net = new_model(device)

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
        val_set, batch_size=args.batch_size,
        collate_fn=collate_fn,
        pin_memory=True
    )

    while True:
        for batch_i, batch in enumerate(train_set_loader):
            logging.info("Loading batch {}".format(batch_i))
            batch = tuple(t.to(device) for t in batch)
            x_state, x_orders, x_power, x_season, y_actions = batch
            logging.info("Starting batch {} of len {}".format(batch_i, len(x_state)))

            optim.zero_grad()
            y_guess = net(x_state, x_orders, x_power, x_season)
            loss = loss_fn(y_guess, y_actions)
            loss.backward()
            optim.step()

            logging.info("batch {} loss={}".format(batch_i, loss))

            # Compute validation accuracy
            if batch_i % 1000 == 0:
                logging.info("Calculating val loss...")
                net.eval()
                losses, batch_sizes, accuracies = [], [], []
                for batch in val_set_loader:
                    x_state, x_orders, x_power, x_season, y_actions = [t.to(device) for t in batch]
                    y_guess = net(x_state, x_orders, x_power, x_season)
                    losses.append(loss_fn(y_guess, y_actions).detach().cpu())
                    batch_sizes.append(len(y_actions))
                    accuracies.append(calculate_accuracy(y_guess, y_actions))
                batch_sizes = [b / sum(batch_sizes) for b in batch_sizes]
                val_loss = sum(l * s for (l, s) in zip(losses, batch_sizes))
                val_accuracy = sum(a * s for (a, s) in zip(accuracies, batch_sizes))
                logging.info("Validation loss={} acc={}".format(val_loss, val_accuracy))
                net.train()

                if args.model_out:
                    logging.info("Saving model to {}".format(args.model_out))
                    torch.save(net.state_dict(), args.model_out)
