#!/usr/bin/env python
import argparse
import atexit
import glob
import logging
import os
import random
import torch
from collections import Counter
from functools import reduce
from torch.utils.data.distributed import DistributedSampler

from fairdiplomacy.data.dataset import Dataset
from fairdiplomacy.models.dipnet.load_model import new_model
from fairdiplomacy.models.dipnet.order_vocabulary import (
    get_order_vocabulary,
    get_order_vocabulary_idxs_by_unit,
    EOS_IDX,
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s"))
logger.addHandler(handler)

ORDER_VOCABULARY = get_order_vocabulary()
ORDER_VOCABULARY_IDXS_BY_UNIT = get_order_vocabulary_idxs_by_unit()


def process_batch(net, batch, loss_fn, temperature=1.0, p_teacher_force=0.0):
    """Calculate a forward pass on a batch

    Returns:
    - loss: the output of loss_fn(logits, targets)
    - order_scores: FIXME FIXME FIXME
    - order_idxs: [B, S] LongTensor of sampled order idxs
    """
    x_state, x_orders, x_power, x_season, x_in_adj_phase, x_possible_actions, x_loc_idxs, y_actions = (
        batch
    )

    # forward pass
    teacher_force_orders = y_actions if torch.rand(1) < p_teacher_force else None
    order_idxs, order_scores = net(
        x_state,
        x_orders,
        x_power,
        x_season,
        x_in_adj_phase,
        x_loc_idxs,
        x_possible_actions,
        temperature=temperature,
        teacher_force_orders=teacher_force_orders,
    )

    # reshape and mask out <EOS> tokens from sequences
    y_actions = y_actions.to(next(net.parameters()).device)
    y_actions = y_actions[:, : order_scores.shape[1]].reshape(-1)  # [B * S]
    try:
        order_scores = order_scores.view(len(y_actions), len(ORDER_VOCABULARY))
    except RuntimeError:
        logger.error(
            f"Bad view: {order_scores.shape} != {len(y_actions)} * {len(ORDER_VOCABULARY)}, {order_idxs.shape}, {order_scores.shape}"
        )
        raise

    order_scores = order_scores[y_actions != EOS_IDX]
    y_actions = y_actions[y_actions != EOS_IDX]

    observed_order_scores = order_scores.gather(1, y_actions.unsqueeze(-1)).squeeze(-1)
    if observed_order_scores.min() < -1e7:
        min_score, min_idx = observed_order_scores.min(0)
        logger.warning(
            f"!!! Got masked order for {get_order_vocabulary()[y_actions[min_idx]]} !!!!"
        )

    # calculate loss
    return loss_fn(order_scores, y_actions), order_scores, order_idxs


def calculate_accuracy(order_idxs, y_truth):
    y_truth = y_truth[: (order_idxs.shape[0]), : (order_idxs.shape[1])].to(order_idxs.device)
    mask = y_truth != EOS_IDX
    return torch.mean((y_truth[mask] == order_idxs[mask]).float())


def calculate_split_accuracy_counts(order_idxs, y_truth):
    counts = Counter()

    y_truth = y_truth[: (order_idxs.shape[0]), : (order_idxs.shape[1])].to(order_idxs.device)
    for b in range(y_truth.shape[0]):
        for s in range(y_truth.shape[1]):
            if y_truth[b, s] == EOS_IDX:
                continue

            truth_order = ORDER_VOCABULARY[y_truth[b, s]]
            correct = y_truth[b, s] == order_idxs[b, s]

            # stats by loc
            loc = truth_order.split()[1]
            counts["loc.{}.{}".format(loc, "y" if correct else "n")] += 1

            # stats by order type
            order_type = truth_order.split()[2]
            counts["type.{}.{}".format(order_type, "y" if correct else "n")] += 1

            # stats by order step
            counts["step.{}.{}".format(s, "y" if correct else "n")] += 1

    return counts


def validate(net, val_set, loss_fn, batch_size):
    with torch.no_grad():
        net.eval()
        batch_losses, batch_accuracies, batch_acc_split_counts = [], [], []

        for batch_idxs in torch.arange(len(val_set)).split(batch_size):
            batch = val_set[batch_idxs]
            y_actions = batch[-1]
            if y_actions.shape[0] == 0:
                logger.warning(
                    "Got an empty validation batch! y_actions.shape={}".format(y_actions.shape)
                )
                continue
            losses, order_scores, order_idxs = process_batch(
                net, batch, loss_fn, temperature=0.001, p_teacher_force=1.0
            )
            batch_losses.append(losses)
            batch_accuracies.append(calculate_accuracy(order_idxs, y_actions))
            batch_acc_split_counts.append(calculate_split_accuracy_counts(order_idxs, y_actions))
        net.train()

    # combine batch losses, accuracies
    val_losses = torch.cat(batch_losses)
    val_loss = torch.mean(val_losses)
    weights = [len(l) / len(val_losses) for l in batch_losses]
    val_accuracy = sum(a * w for (a, w) in zip(batch_accuracies, weights))

    # combine accuracy splits
    split_counts = reduce(
        lambda x, y: Counter({k: x[k] + y[k] for k in set(x.keys()) | set(y.keys())}),
        batch_acc_split_counts,
        Counter(),
    )
    split_pcts = {
        k: split_counts[k + ".y"] / (split_counts[k + ".y"] + split_counts[k + ".n"])
        for k in [k.rsplit(".", 1)[0] for k in split_counts.keys()]
    }

    return val_loss, val_accuracy, split_pcts


def main_subproc(rank, world_size, args, train_set, val_set):
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

    logger.info("Init model...")
    net = new_model(args)

    # send model to GPU
    logger.debug("net.cuda({})".format(rank))
    net.cuda(rank)
    logger.debug("net {} DistributedDataParallel".format(rank))
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[rank])
    logger.debug("net {} DistributedDataParallel done".format(rank))

    # load from checkpoint if specified
    if checkpoint:
        logger.debug("net.load_state_dict")
        net.load_state_dict(checkpoint["model"], strict=True)

    # create optimizer, from checkpoint if specified
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    if checkpoint:
        optim.load_state_dict(checkpoint["optim"])

    train_set_sampler = DistributedSampler(train_set)
    for epoch in range(checkpoint["epoch"] + 1 if checkpoint else 0, 10000):
        train_set_sampler.set_epoch(epoch)
        batches = torch.tensor(list(iter(train_set_sampler)), dtype=torch.long).split(
            args.batch_size
        )
        for batch_i, batch_idxs in enumerate(batches):
            batch = train_set[batch_idxs]
            logger.debug(f"Zero grad {batch_i} ...")

            # check batch is not empty
            if (batch[-1] == EOS_IDX).all():
                logger.warning("Skipping empty epoch {} batch {}".format(epoch, batch_i))
                continue

            # learn
            logger.info("Starting epoch {} batch {}".format(epoch, batch_i))
            optim.zero_grad()
            losses, order_scores, order_idxs = process_batch(
                net, batch, loss_fn, p_teacher_force=args.teacher_force
            )

            loss = torch.mean(losses)
            loss.backward()
            logger.debug(f"Running step {batch_i} ...")
            optim.step()
            if rank == 0:
                logger.info(f"epoch {epoch} batch {batch_i} / {len(batches)} loss= {loss}")

            # calculate validation loss/accuracy
            if (
                not args.skip_validation
                and (epoch * len(batches) + batch_i) % args.validate_every == 0
            ):
                if rank == 0:
                    logger.info("Calculating val loss...")
                val_loss, val_accuracy, split_pcts = validate(
                    net, val_set, loss_fn, args.batch_size
                )

                if rank == 0:
                    logger.info(
                        f"Validation epoch= {epoch} batch= {batch_i} loss= {val_loss} acc= {val_accuracy}"
                    )
                    for k, v in sorted(split_pcts.items()):
                        logger.info(f"val split epoch= {epoch} batch= {batch_i}: {k} = {v}")

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
                            "args": args,
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
    parser.add_argument(
        "--data-dir",
        default="/checkpoint/jsgray/diplomacy/mila_dataset/data",
        help="Path to dir containing game.json files",
    )
    parser.add_argument("--data-cache", help="Path to dir containing dataset cache")
    parser.add_argument(
        "--num-dataloader-workers",
        type=int,
        default=20,
        help="Dataloader procs (1 means load in the main process)",
    )
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--checkpoint", default="./checkpoint.pth", help="Path to load/save the model"
    )
    parser.add_argument(
        "--val-set-pct", type=float, default=0.01, help="Percentage of games to use as val set"
    )
    parser.add_argument(
        "--teacher-force", type=float, default=1.0, help="Prob[teacher forcing] during training"
    )
    parser.add_argument("--lstm-dropout", type=float, default=0, help="LSTM dropout pct")
    parser.add_argument(
        "--debug-only-opening-phase", action="store_true", help="If set, restrict data to S1901M"
    )
    parser.add_argument("--debug-no-mp", action="store_true", help="If set, use a single process")
    parser.add_argument(
        "--validate-every", type=int, default=1000, help="Validate/save every # of batches"
    )
    parser.add_argument("--skip-validation", action="store_true", help="Skip validation / save")
    parser.add_argument("--learnable-A", action="store_true", help="Learn adjacency matrix")
    parser.add_argument(
        "--learnable-alignments", action="store_true", help="Learn attention alignment matrix"
    )
    parser.add_argument(
        "--avg-embedding",
        action="store_true",
        help="Average across location embedding instead of using attention",
    )
    args = parser.parse_args()
    logger.warning("Args: {}, file={}".format(args, os.path.abspath(__file__)))

    n_gpus = torch.cuda.device_count()
    logger.info("Using {} GPUs".format(n_gpus))

    # search for data and create train/val splits
    if args.data_cache and os.path.exists(args.data_cache):
        logger.info(f"Found dataset cache at {args.data_cache}")
        train_dataset, val_dataset = torch.load(args.data_cache)
    else:
        assert args.data_dir is not None
        game_jsons = glob.glob(os.path.join(args.data_dir, "*.json"))
        assert len(game_jsons) > 0
        logger.info(f"Found dataset of {len(game_jsons)} games...")
        val_game_jsons = random.sample(game_jsons, int(len(game_jsons) * args.val_set_pct))
        train_game_jsons = list(set(game_jsons) - set(val_game_jsons))

        train_dataset = Dataset(train_game_jsons, n_jobs=args.num_dataloader_workers)
        val_dataset = Dataset(val_game_jsons, n_jobs=args.num_dataloader_workers)
        if args.data_cache:
            logger.info(f"Saving datasets to {args.data_cache}")
            torch.save((train_dataset, val_dataset), args.data_cache)

    logger.info(f"Train dataset: {train_dataset.stats_str()}")
    logger.info(f"Val dataset: {val_dataset.stats_str()}")

    # required when using multithreaded DataLoader
    try:
        torch.multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    if args.debug_no_mp:
        main_subproc(0, 1, args, train_dataset, val_dataset)
    else:
        torch.multiprocessing.spawn(
            main_subproc, nprocs=n_gpus, args=(n_gpus, args, train_dataset, val_dataset)
        )
