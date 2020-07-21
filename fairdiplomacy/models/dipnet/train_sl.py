#!/usr/bin/env python
import atexit
import json
import logging
import os
import random
import torch
from collections import Counter
from functools import reduce
from torch.utils.data.distributed import DistributedSampler

from fairdiplomacy.data.dataset import Dataset, DataFields
from fairdiplomacy.models.dipnet.load_model import new_model
from fairdiplomacy.models.consts import POWERS
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
logger.propagate = False

ORDER_VOCABULARY = get_order_vocabulary()
ORDER_VOCABULARY_IDXS_BY_UNIT = get_order_vocabulary_idxs_by_unit()


def process_batch(
    net,
    batch,
    policy_loss_fn,
    value_loss_fn,
    temperature=1.0,
    p_teacher_force=1.0,
    shuffle_locs=False,
):
    """Calculate a forward pass on a batch

    Returns:
    - policy_losses: [?] FloatTensor, unknown size due to unknown # of non-zero actions
    - value_losses: [B] FloatTensor
    - sampled_idxs: [B, S] LongTensor of sampled order idxs (< 469)
    - final_sos: [B, 7] estimated final sum-of-squares share of each power
    """
    assert p_teacher_force == 1
    device = next(net.parameters()).device

    if shuffle_locs:
        y_actions = batch["y_actions"]
        B, L = y_actions.shape

        loc_priority = torch.rand(B, L)
        loc_priority += (y_actions == -1) * 1000
        perm = loc_priority.sort(dim=-1).indices

        batch["y_actions"] = y_actions.gather(-1, perm)

        batch["x_possible_actions"] = batch["x_possible_actions"].gather(
            -2, perm.unsqueeze(-1).repeat(1, 1, 469)
        )

        # x_loc_idxs is B x 81, where the value in each loc is which order in
        # the sequence it is (or -1 if not in the sequence)
        new_x_loc_idxs = batch["x_loc_idxs"].clone()
        for lidx in range(L):
            mask = batch["x_loc_idxs"] == perm[:, lidx].unsqueeze(-1)
            new_x_loc_idxs[mask] = lidx
        batch["x_loc_idxs"] = new_x_loc_idxs

    # forward pass
    teacher_force_orders = (
        cand_idxs_to_order_idxs(batch["y_actions"], batch["x_possible_actions"], pad_out=0)
        if torch.rand(1) < p_teacher_force
        else None
    )
    order_idxs, sampled_idxs, logits, final_sos = net(
        **{k: v for k, v in batch.items() if k.startswith("x_")},
        temperature=temperature,
        teacher_force_orders=teacher_force_orders,
    )

    # x_possible_actions = batch['x_possible_actions'].to(device)
    y_actions = batch["y_actions"].to(device)

    # reshape and mask out <EOS> tokens from sequences
    y_actions = y_actions[:, : logits.shape[1]].reshape(-1)  # [B * S]
    try:
        logits = logits.view(len(y_actions), -1)
    except RuntimeError:
        logger.error(f"Bad view: {logits.shape}, {order_idxs.shape}, {y_actions.shape}")
        raise

    logits = logits[y_actions != EOS_IDX]
    y_actions = y_actions[y_actions != EOS_IDX]

    observed_logits = logits.gather(1, y_actions.unsqueeze(-1)).squeeze(-1)
    if observed_logits.min() < -1e7:
        min_score, min_idx = observed_logits.min(0)
        logger.warning(
            f"!!! Got masked order for {get_order_vocabulary()[y_actions[min_idx]]} !!!"
        )

    # calculate policy loss
    policy_loss = policy_loss_fn(logits, y_actions)

    # calculate sum-of-squares value loss
    y_final_scores = batch["y_final_scores"].to(device).float().squeeze(1)
    value_loss = value_loss_fn(final_sos, y_final_scores)

    # a given state appears multiple times in the dataset for different powers,
    # but we always compute the value loss for each power. So we need to reweight
    # the value loss by 1/num_valid_powers
    value_loss /= batch["valid_power_idxs"].sum(-1, keepdim=True).to(device)

    return policy_loss, value_loss, sampled_idxs, final_sos


def cand_idxs_to_order_idxs(idxs, candidates, pad_out=EOS_IDX):
    """Convert from idxs in candidates to idxs in ORDER_VOCABULARY

    Arguments:
    - idxs: [B, S] candidate idxs, each 0 - 469, padding=EOS_IDX
    - candidates: [B, S, 469] order idxs of each candidate, 0 - 13k

    Return [B, S] of order idxs, 0 - 13k, padding=pad_out
    """
    mask = idxs.view(-1) != EOS_IDX
    flat_candidates = candidates.view(-1, candidates.shape[2])
    r = torch.empty_like(idxs).fill_(pad_out).view(-1)
    r[mask] = flat_candidates[mask].gather(1, idxs.view(-1)[mask].unsqueeze(1)).view(-1)
    return r.view(*idxs.shape)


def calculate_accuracy(sampled_idxs, y_truth):
    y_truth = y_truth[: (sampled_idxs.shape[0]), : (sampled_idxs.shape[1])].to(sampled_idxs.device)
    mask = y_truth != EOS_IDX
    return torch.mean((y_truth[mask] == sampled_idxs[mask]).float())


def calculate_value_accuracy(final_sos, y_final_scores):
    """Return top-1 accuracy"""
    y_final_scores = y_final_scores.squeeze(1)
    actual_winner = y_final_scores == y_final_scores.max(dim=1, keepdim=True).values
    guessed_winner = final_sos == final_sos.max(dim=1, keepdim=True).values
    return (actual_winner & guessed_winner).any(dim=1).float().mean()


def calculate_split_accuracy_counts(sampled_idxs, y_truth):
    counts = Counter()

    y_truth = y_truth[: (sampled_idxs.shape[0]), : (sampled_idxs.shape[1])].to(sampled_idxs.device)
    for b in range(y_truth.shape[0]):
        for s in range(y_truth.shape[1]):
            if y_truth[b, s] == EOS_IDX:
                continue

            truth_order = ORDER_VOCABULARY[y_truth[b, s]]
            correct = y_truth[b, s] == sampled_idxs[b, s]

            # stats by loc
            loc = truth_order.split()[1]
            counts["loc.{}.{}".format(loc, "y" if correct else "n")] += 1

            # stats by order type
            order_type = truth_order.split()[2]
            counts["type.{}.{}".format(order_type, "y" if correct else "n")] += 1

            # stats by order step
            counts["step.{}.{}".format(s, "y" if correct else "n")] += 1

    return counts


def validate(net, val_set, policy_loss_fn, value_loss_fn, batch_size, value_loss_weight: float):
    net_device = next(net.parameters()).device

    with torch.no_grad():
        net.eval()

        batch_losses = []
        batch_accuracies = []
        batch_acc_split_counts = []
        batch_value_accuracies = []

        for batch_idxs in torch.arange(len(val_set)).split(batch_size):
            batch = val_set[batch_idxs]
            batch = DataFields({k: v.to(net_device) for k, v in batch.items()})
            y_actions = batch["y_actions"]
            if y_actions.shape[0] == 0:
                logger.warning(
                    "Got an empty validation batch! y_actions.shape={}".format(y_actions.shape)
                )
                continue
            policy_losses, value_losses, sampled_idxs, final_sos = process_batch(
                net, batch, policy_loss_fn, value_loss_fn, temperature=0.001, p_teacher_force=1.0
            )

            batch_losses.append((policy_losses, value_losses))
            batch_accuracies.append(calculate_accuracy(sampled_idxs, y_actions))
            batch_value_accuracies.append(
                calculate_value_accuracy(final_sos, batch["y_final_scores"])
            )
            batch_acc_split_counts.append(
                calculate_split_accuracy_counts(
                    sampled_idxs,
                    cand_idxs_to_order_idxs(
                        batch["y_actions"], batch["x_possible_actions"], pad_out=EOS_IDX
                    ),
                )
            )
        net.train()

    # validation loss
    p_losses, v_losses = [torch.cat(x) for x in zip(*batch_losses)]
    p_loss = torch.mean(p_losses)
    v_loss = torch.mean(v_losses)
    valid_loss = (1 - value_loss_weight) * p_loss + value_loss_weight * v_loss

    # validation accuracy
    weights = [len(pl) / len(p_losses) for pl, _ in batch_losses]
    valid_p_accuracy = sum(a * w for (a, w) in zip(batch_accuracies, weights))
    valid_v_accuracy = sum(a * w for (a, w) in zip(batch_value_accuracies, weights))

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

    return valid_loss, p_loss, v_loss, valid_p_accuracy, valid_v_accuracy, split_pcts


def maybe_build_jsonl_logger(fpath, do_log):
    """Returns a function that either dumps scalar values to a file or noop.

    Returned function has signature:

        def log(**kwargs):
            ....

    If do_log is False, then the function is doing nothing. Otherwise, the
    function expectes to get a dict of json-serializable values and will
    append them to fpath as a jsonl.
    """
    stream = open(fpath, "a") if do_log else None

    def sanitize(value):
        if hasattr(value, "item"):
            return value.item()
        return value

    def log(**kwargs):
        if stream is not None:
            record = {k: sanitize(v) for k, v in kwargs.items()}
            print(json.dumps(record), file=stream, flush=True)

    return log


def main_subproc(rank, world_size, args, train_set, val_set):
    # distributed training setup
    mp_setup(rank, world_size)
    atexit.register(mp_cleanup)
    torch.cuda.set_device(rank)

    write_jsonl = rank == 0 and getattr(args, "write_jsonl", False)
    log_scalars = maybe_build_jsonl_logger("metrics.jsonl", write_jsonl)

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
    policy_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    value_loss_fn = torch.nn.MSELoss(reduction="none")
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=args.lr_decay)
    if checkpoint:
        optim.load_state_dict(checkpoint["optim"])

    best_loss, best_p_loss, best_v_loss = None, None, None

    train_set_sampler = DistributedSampler(train_set)
    for epoch in range(checkpoint["epoch"] + 1 if checkpoint else 0, args.num_epochs):
        train_set_sampler.set_epoch(epoch)
        batches = torch.tensor(list(iter(train_set_sampler)), dtype=torch.long).split(
            args.batch_size
        )
        for batch_i, batch_idxs in enumerate(batches):
            batch = train_set[batch_idxs]
            logger.debug(f"Zero grad {batch_i} ...")

            # check batch is not empty
            if (batch["y_actions"] == EOS_IDX).all():
                logger.warning("Skipping empty epoch {} batch {}".format(epoch, batch_i))
                continue

            # learn
            logger.debug("Starting epoch {} batch {}".format(epoch, batch_i))
            optim.zero_grad()
            policy_losses, value_losses, _, _ = process_batch(
                net,
                batch,
                policy_loss_fn,
                value_loss_fn,
                p_teacher_force=args.teacher_force,
                shuffle_locs=args.shuffle_locs,
            )

            # backward
            p_loss = torch.mean(policy_losses)
            v_loss = torch.mean(value_losses)
            loss = (1 - args.value_loss_weight) * p_loss + args.value_loss_weight * v_loss
            loss.backward()

            # clip gradients, step
            value_decoder_grad_norm = torch.nn.utils.clip_grad_norm_(
                getattr(net, "module", net).value_decoder.parameters(),
                args.value_decoder_clip_grad_norm,
            )
            grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip_grad_norm)
            optim.step()

            # log diagnostics
            if rank == 0 and batch_i % 10 == 0:
                scalars = dict(
                    epoch=epoch,
                    batch=batch_i,
                    loss=loss,
                    lr=optim.state_dict()["param_groups"][0]["lr"],
                    grad_norm=grad_norm,
                    value_decoder_grad_norm=value_decoder_grad_norm,
                    p_loss=p_loss,
                    v_loss=v_loss,
                )
                log_scalars(**scalars)
                logger.info(
                    "epoch {} batch {} / {}, ".format(epoch, batch_i, len(batches))
                    + " ".join(f"{k}= {v}" for k, v in scalars.items())
                )

        # calculate validation loss/accuracy
        if not args.skip_validation and rank == 0:
            logger.info("Calculating val loss...")
            (
                valid_loss,
                valid_p_loss,
                valid_v_loss,
                valid_p_accuracy,
                valid_v_accuracy,
                split_pcts,
            ) = validate(
                net,
                val_set,
                policy_loss_fn,
                value_loss_fn,
                args.batch_size,
                value_loss_weight=args.value_loss_weight,
            )
            scalars = dict(
                epoch=epoch,
                valid_loss=valid_loss,
                valid_p_loss=valid_p_loss,
                valid_v_loss=valid_v_loss,
                valid_p_accuracy=valid_p_accuracy,
                valid_v_accuracy=valid_v_accuracy,
            )
            log_scalars(**scalars)
            logger.info("Validation " + " ".join([f"{k}= {v}" for k, v in scalars.items()]))
            for k, v in sorted(split_pcts.items()):
                logger.info(f"val split epoch= {epoch} batch= {batch_i}: {k} = {v}")

            # save model
            if args.checkpoint and rank == 0:
                obj = {
                    "model": net.state_dict(),
                    "optim": optim.state_dict(),
                    "epoch": epoch,
                    "batch_i": batch_i,
                    "valid_p_accuracy": valid_p_accuracy,
                    "args": args,
                }
                logger.info("Saving checkpoint to {}".format(args.checkpoint))
                torch.save(obj, args.checkpoint)

                if best_loss is None or valid_loss < best_loss:
                    torch.save(obj, args.checkpoint + ".best")
                if best_p_loss is None or valid_p_loss < best_p_loss:
                    torch.save(obj, args.checkpoint + ".bestp")
                if best_v_loss is None or valid_v_loss < best_v_loss:
                    torch.save(obj, args.checkpoint + ".bestv")

        lr_scheduler.step()


def mp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.manual_seed(0)
    random.seed(0)


def mp_cleanup():
    torch.distributed.destroy_process_group()


def run_with_cfg(args):
    random.seed(0)

    logger.warning("Args: {}, file={}".format(args, os.path.abspath(__file__)))

    n_gpus = torch.cuda.device_count()
    logger.info("Using {} GPUs".format(n_gpus))

    # search for data and create train/val splits
    if args.data_cache and os.path.exists(args.data_cache):
        logger.info(f"Found dataset cache at {args.data_cache}")
        train_dataset, val_dataset = torch.load(args.data_cache)
    else:
        assert args.metadata_path is not None
        assert args.data_dir is not None
        game_metadata, min_rating, train_game_ids, val_game_ids = get_db_cache_args(args)

        train_dataset = Dataset(
            game_ids=train_game_ids,
            data_dir=args.data_dir,
            game_metadata=game_metadata,
            only_with_min_final_score=args.only_with_min_final_score,
            n_jobs=args.num_dataloader_workers,
            value_decay_alpha=args.value_decay_alpha,
            min_rating=min_rating,
            exclude_n_holds=args.exclude_n_holds,
        )
        val_dataset = Dataset(
            game_ids=val_game_ids,
            data_dir=args.data_dir,
            game_metadata=game_metadata,
            only_with_min_final_score=args.only_with_min_final_score,
            n_jobs=args.num_dataloader_workers,
            value_decay_alpha=args.value_decay_alpha,
            min_rating=min_rating,
            exclude_n_holds=args.exclude_n_holds,
        )
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


def get_db_cache_args(args):
    with open(args.metadata_path) as meta_f:
        game_metadata = json.load(meta_f)
    # convert to int game keys
    game_metadata = {int(k): v for k, v in game_metadata.items()}
    game_ids = list(game_metadata.keys())
    # compute min rating
    if args.min_rating_percentile > 0:
        ratings = torch.tensor(
            [
                game[pwr]["logit_rating"]
                for game in game_metadata.values()
                for pwr in POWERS
                if pwr in game
            ]
        )
        min_rating = ratings.sort()[0][int(len(ratings) * args.min_rating_percentile)]
        print(
            f"Only training on games with min rating of {min_rating} ({args.min_rating_percentile * 100} percentile)"
        )
    else:
        min_rating = -1e9
    if args.max_games > 0:
        game_ids = game_ids[: args.max_games]
    assert len(game_ids) > 0
    logger.info(f"Found dataset of {len(game_ids)} games...")
    val_game_ids = random.sample(game_ids, max(1, int(len(game_ids) * args.val_set_pct)))
    train_game_ids = list(set(game_ids) - set(val_game_ids))

    return game_metadata, min_rating, train_game_ids, val_game_ids
