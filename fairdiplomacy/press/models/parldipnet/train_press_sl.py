#!/usr/bin/env python

from fairdiplomacy.models.dipnet.train_sl import *
from fairdiplomacy.press.data.dataset import build_press_dataset
from fairdiplomacy.press.models.parldipnet.parldipnet import ParlaiEncoderDipNet
from parlai.core.agents import create_agent_from_model_file

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s"))
logger.addHandler(handler)
logger.propagate = False


# Adapted from train_sl - Function will very likely change in the near future.
def main_subproc(rank, world_size, args, train_set, val_set):
    # distributed training setup
    mp_setup(rank, world_size)
    atexit.register(mp_cleanup)
    torch.cuda.set_device(rank)

    metric_logger = Logger(is_master=rank == 0)
    global_step = 0
    log_scalars = lambda **scalars: metric_logger.log_metrics(
        scalars, step=global_step, sanitize=True
    )

    logger.info("Init model...")
    net = ParlaiEncoderDipNet(
        no_dialogue_emb=args.no_dialogue_emb,
        encoder_model_path=args.encoder_model_path,
        dipnet_args=args.dipnet_train_params,
        combine_emb_size=args.combine_emb_size,
        combine_num_layers=args.combine_num_layers,
    )

    args = args.dipnet_train_params

    # send model to GPU
    logger.debug("net.cuda({})".format(rank))
    net.cuda(rank)
    logger.debug("net {} DistributedDataParallel".format(rank))
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[rank])
    logger.debug("net {} DistributedDataParallel done".format(rank))

    # create optimizer, from checkpoint if specified
    policy_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    value_loss_fn = torch.nn.MSELoss(reduction="none")
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=args.lr_decay)

    best_loss, best_p_loss, best_v_loss = None, None, None

    train_set_sampler = DistributedSampler(train_set)
    for epoch in range(0, args.num_epochs):
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
                getattr(net, "module", net).dipnet.value_decoder.parameters(),
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
            global_step += 1

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


def run_with_cfg(args):
    random.seed(0)

    logger.warning("Args: {}, file={}".format(args, os.path.abspath(__file__)))
    n_gpus = torch.cuda.device_count()
    logger.info("Using {} GPUs".format(n_gpus))

    data_cache = args.dipnet_train_params.data_cache
    if data_cache and os.path.exists(data_cache):
        logger.info(f"Found press dataset cache at {data_cache}")
        train_dataset, val_dataset = torch.load(data_cache)
    else:
        no_press_cfg = args.dipnet_train_params.dataset_params
        game_metadata, min_rating, train_game_ids, val_game_ids = get_sl_db_args(
            args.dipnet_train_params.metadata_path,
            args.dipnet_train_params.min_rating_percentile,
            args.dipnet_train_params.max_games,
            args.dipnet_train_params.val_set_pct,
        )

        train_dataset, val_dataset = build_press_dataset(
            game_metadata, min_rating, no_press_cfg, args, train_game_ids, val_game_ids
        )

        logger.info(f"Saving press datasets to {data_cache}")
        torch.save((train_dataset, val_dataset), data_cache)

    logger.info(f"Train dataset: {train_dataset.stats_str()}")
    logger.info(f"Val dataset: {val_dataset.stats_str()}")

    if not os.path.exists(args.encoder_model_path):
        save_parlai_encoder(args.parlai_agent_file, args.encoder_model_path)

    # required when using multithreaded DataLoader
    try:
        torch.multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    if args.dipnet_train_params.debug_no_mp:
        main_subproc(0, 1, args, train_dataset, val_dataset)
    else:
        logger.info("Using multiprocessing...")
        torch.multiprocessing.spawn(
            main_subproc, nprocs=n_gpus, args=(n_gpus, args, train_dataset, val_dataset),
        )


def save_parlai_encoder(parlai_agent_file, encoder_model_path):
    """
    Saves parlai_agent's encoder to encode_model_path
    :param parlai_agent_file:
    :param encoder_model_path:
    :return:
    """
    logging.info("Loading parlai agent...")
    dialogue_model = create_agent_from_model_file(
        parlai_agent_file,
        opt_overides={"fp16": False, "data_parallel": False, "model_parallel": False,},
    ).model
    torch.save(dialogue_model.module.encoder, encoder_model_path)
