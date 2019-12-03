import argparse
import atexit
import glob
import logging
import os
import random
import torch
from torch.utils.data.distributed import DistributedSampler

from fairdiplomacy.data.dataset import Dataset, collate_fn
from fairdiplomacy.data.get_game_lengths import get_all_game_lengths
from fairdiplomacy.models.consts import ADJACENCY_MATRIX
from fairdiplomacy.models.dipnet.dipnet import SimpleDipNet as DipNet
from fairdiplomacy.models.dipnet.order_vocabulary import (
    get_order_vocabulary,
    get_order_vocabulary_idxs_by_unit,
    EOS_IDX,
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s"))
logger.addHandler(handler)

BOARD_STATE_SIZE = 35
PREV_ORDERS_SIZE = 40
INTER_EMB_SIZE = 120
POWER_EMB_SIZE = 7
SEASON_EMB_SIZE = 3
NUM_ENCODER_BLOCKS = 16
LSTM_SIZE = 200
ORDER_EMB_SIZE = 80

ORDER_VOCABULARY = get_order_vocabulary()
ORDER_VOCABULARY_IDXS_BY_UNIT = get_order_vocabulary_idxs_by_unit()


def new_model():
    A = torch.from_numpy(ADJACENCY_MATRIX).float()
    return DipNet(
        BOARD_STATE_SIZE,
        PREV_ORDERS_SIZE,
        INTER_EMB_SIZE,
        POWER_EMB_SIZE,
        SEASON_EMB_SIZE,
        NUM_ENCODER_BLOCKS,
        A,
        len(ORDER_VOCABULARY),
        # LSTM_SIZE,
        # ORDER_EMB_SIZE,
    )


def process_batch(net, batch, loss_fn):
    """Calculate a forward pass on a batch

    Returns:
    - loss: the output of loss_fn(logits, targets)
    - order_idxs: [B, S] LongTensor of sampled order idxs
    """
    x_state, x_orders, x_power, x_season, y_actions = batch
    order_mask = build_order_mask(y_actions)

    # forward pass
    order_idxs, order_scores = net(x_state, x_orders, x_power, x_season, order_mask)

    # reshape and mask out <EOS> tokens from sequences
    y_actions = y_actions.to(next(net.parameters()).device)
    y_actions = y_actions[:, : order_idxs.shape[1]].reshape(-1)  # [B * S]
    order_scores = order_scores.view(len(y_actions), len(ORDER_VOCABULARY))
    order_scores = order_scores[y_actions != EOS_IDX]
    y_actions = y_actions[y_actions != EOS_IDX]

    # calculate loss
    return loss_fn(order_scores, y_actions), order_idxs


def build_order_mask(y_actions):
    """Build a mask of valid possible actions given a ground truth sequence

    Arguments:
    - y_actions: [B, 17] LongTensor of order idxs

    Returns:
    - [B, S, 13k] BoolTensor mask, 0 < S <= 17, where idx [b, s, x] == True if
      ORDER_VOCABULARY[x] is an order originating from the same unit as y_actions[b, s]
    """
    order_mask = torch.zeros(
        y_actions.shape[0], y_actions.shape[1], len(ORDER_VOCABULARY), dtype=torch.bool
    )
    for step in range(order_mask.shape[1]):
        if (y_actions[:, step] == EOS_IDX).all():
            break
        for b in range(order_mask.shape[0]):
            order_idx = y_actions[b, step]
            if order_idx == EOS_IDX:
                continue
            order = ORDER_VOCABULARY[order_idx]
            unit = " ".join(order.split()[:2])
            try:
                valid_idxs = ORDER_VOCABULARY_IDXS_BY_UNIT[unit]
            except KeyError:
                valid_idxs = ORDER_VOCABULARY_IDXS_BY_UNIT[unit.split("/")[0]]
            order_mask[b, step, valid_idxs] = 1

    return order_mask[:, :step, :]


def calculate_accuracy(order_idxs, y_truth):
    y_truth = y_truth[: (order_idxs.shape[0]), : (order_idxs.shape[1])].to(order_idxs.device)
    mask = y_truth != EOS_IDX
    return torch.mean((y_truth[mask] == order_idxs[mask]).float())


def validate(net, val_set_loader, loss_fn):
    with torch.no_grad():
        net.eval()
        batch_losses, batch_accuracies = [], []
        for batch in val_set_loader:
            _, _, _, _, y_actions = batch
            losses, order_idxs = process_batch(net, batch, loss_fn)
            batch_losses.append(losses)
            batch_accuracies.append(calculate_accuracy(order_idxs, y_actions))
        net.train()

    val_losses = torch.cat(batch_losses)
    val_loss = torch.mean(val_losses)
    weights = [len(l) / len(val_losses) for l in batch_losses]
    val_accuracy = sum(a * w for (a, w) in zip(batch_accuracies, weights))

    return val_loss, val_accuracy


def main_subproc(
    rank,
    world_size,
    args,
    train_game_jsons,
    train_game_json_lengths,
    val_game_jsons,
    val_game_json_lengths,
):
    # distributed training setup
    mp_setup(rank, world_size)
    atexit.register(mp_cleanup)
    torch.cuda.set_device(rank)

    # load checkpoint if specified
    if args.checkpoint and os.path.isfile(args.checkpoint):
        logger.info("Loading checkpoint at {}".format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint, map_location="cuda:{}".format(rank))
    else:
        checkpoint = None

    # create model, from checkpoint if specified
    logger.info("Init model...")
    net = new_model()
    if checkpoint:
        logger.debug("net.load_state_dict")
        net.load_state_dict(checkpoint["model"], strict=False)

    # send model to GPU
    logger.debug("net.cuda({})".format(rank))
    net.cuda(rank)
    logger.debug("net {} DistributedDataParallel".format(rank))
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[rank])
    logger.debug("net {} DistributedDataParallel done".format(rank))

    # create optimizer, from checkpoint if specified
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    if checkpoint:
        optim.load_state_dict(checkpoint["optim"])

    # Create datasets / loaders
    train_set = Dataset(train_game_jsons, train_game_json_lengths)
    val_set = Dataset(val_game_jsons, val_game_json_lengths)
    train_set_sampler = DistributedSampler(train_set)
    train_set_loader = torch.utils.data.DataLoader(
        train_set,
        num_workers=args.num_dataloader_workers,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        pin_memory=True,
        sampler=train_set_sampler,
    )
    val_set_loader = torch.utils.data.DataLoader(
        val_set,
        num_workers=args.num_dataloader_workers,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    for epoch in range(checkpoint["epoch"] + 1 if checkpoint else 0, 10000):
        train_set_sampler.set_epoch(epoch)

        for batch_i, batch in enumerate(train_set_loader):
            # check batch is not empty
            if (batch[-1] == EOS_IDX).all():
                logger.warning("Skipping empty epoch {} batch {}".format(epoch, batch_i))
                continue

            # learn
            logger.info("Starting epoch {} batch {}".format(epoch, batch_i))
            optim.zero_grad()
            losses, _ = process_batch(net, batch, loss_fn)
            loss = torch.mean(losses)
            logger.debug("hi mom start backward")
            loss.backward()
            optim.step()
            logger.info("epoch {} batch {} loss={}".format(epoch, batch_i, loss))

            # calculate validation loss/accuracy
            if (epoch * len(train_set_loader) + batch_i) % 1000 == 0:
                logger.info("Calculating val loss...")
                val_loss, val_accuracy = validate(net, val_set_loader, loss_fn)
                logger.info(
                    "Validation epoch={} batch={} loss={} acc={}".format(
                        epoch, batch_i, val_loss, val_accuracy
                    )
                )

                # save model
                if args.checkpoint and rank == 0:
                    logger.info("Saving checkpoint to {}".format(args.checkpoint))
                    torch.save(
                        {
                            "model": net.state_dict(),
                            "optim": optim.state_dict(),
                            "epoch": epoch,
                            "batch_i": batch_i,
                            "val_accuracy": val_accuracy,
                        },
                        args.checkpoint,
                    )


def mp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.manual_seed(0)
    random.seed(0)


def mp_cleanup():
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Path to dir containing game.json files")
    parser.add_argument(
        "--num-dataloader-workers", type=int, default=32, help="# Dataloader procs"
    )
    parser.add_argument("--batch-size", type=int, default=200, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--checkpoint", help="Path to load/save the model")
    parser.add_argument(
        "--val-set-pct", type=float, default=0.01, help="Percentage of games to use as val set"
    )
    args = parser.parse_args()
    logger.warning("Args: {}".format(args))

    n_gpus = torch.cuda.device_count()
    logger.info("Using {} GPUs".format(n_gpus))

    # search for data and create train/val splits
    game_jsons = glob.glob(os.path.join(args.data_dir, "*.json"))
    assert len(game_jsons) > 0
    logger.info("Found dataset of {} games...".format(len(game_jsons)))
    val_game_jsons = random.sample(game_jsons, int(len(game_jsons) * args.val_set_pct))
    train_game_jsons = list(set(game_jsons) - set(val_game_jsons))

    # Calculate dataset sizes
    logger.info("Calculating dataset sizes")
    train_game_json_lengths = get_all_game_lengths(train_game_jsons)
    val_game_json_lengths = get_all_game_lengths(val_game_jsons)

    # required when using multithreaded DataLoader
    try:
        torch.multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    torch.multiprocessing.spawn(
        main_subproc,
        nprocs=n_gpus,
        args=(
            n_gpus,
            args,
            train_game_jsons,
            train_game_json_lengths,
            val_game_jsons,
            val_game_json_lengths,
        ),
    )
