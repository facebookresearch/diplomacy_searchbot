# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functools
import logging
from typing import Any, Dict, Optional, Tuple

import torch

import heyhi
from fairdiplomacy.models.consts import ADJACENCY_MATRIX, MASTER_ALIGNMENTS
from fairdiplomacy.models.diplomacy_model.order_vocabulary import get_order_vocabulary
from fairdiplomacy.models.diplomacy_model.diplomacy_model import DiplomacyModel
import conf.conf_cfgs


CACHE_SIZE = 5

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


def new_model(args: conf.conf_cfgs.TrainTask) -> DiplomacyModel:
    return DiplomacyModel(
        board_state_size=BOARD_STATE_SIZE,
        # prev_orders_size=PREV_ORDERS_SIZE,
        prev_order_emb_size=PREV_ORDER_EMB_SIZE,
        inter_emb_size=getattr(args, "inter_emb_size", INTER_EMB_SIZE),
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
        use_simple_alignments=args.use_simple_alignments,
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
        separate_value_encoder=args.separate_value_encoder,
        use_global_pooling=args.use_global_pooling,
        encoder_cfg=args.encoder,
        pad_spatial_size_to_multiple=args.pad_spatial_size_to_multiple,
        all_powers=getattr(args, "all_powers", False),
        has_policy=args.has_policy,
        has_value=args.has_value,
    )


def load_diplomacy_model_model_and_args(
    checkpoint_path,
    map_location: str = "cpu",
    eval: bool = False,
    override_has_policy: Optional[bool] = None,
    override_has_value: Optional[bool] = None,
) -> Tuple[DiplomacyModel, Dict[str, Any]]:
    """Load a diplomacy_model model and its args, which should be a dict of the TrainTask prototxt.

    If override_has_policy or override_has_value are left as None, then the default from the model checkpoint will be used.
    If explicitly given as True, then we will fail immediately if the model does not support that output instead of at inference.
    If explicitly given as False, then we will omit loading any unnecessary model parameters, and the model will

    Args:
        checkpoint_path: File path to load from, e.g. my_model.ckpt
        map_location: Device string like "cpu" or "cuda" or "cuda:0" to load the model tensors on.
        eval: If true, set the model into inference mode.
        override_has_policy: Override the "has_policy" field in the TrainTask args.
        override_has_value: Override the "has_value" field in the TrainTask args.

    Returns:
        (diplomacy_model_model, dictified version of conf.conf_cfgs.TrainTask)
    """
    if map_location != "cpu" and not torch.cuda.is_available():
        logging.warning("No CUDA so will load model to CPU instead of %s", map_location)
        map_location = "cpu"

    # Loading model to gpu right away will load optimizer state we don't care about.
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    args = checkpoint["args"]
    if not isinstance(args, dict):
        args = heyhi.conf_to_dict(args)

    load_state_dict_strict = True
    if override_has_policy is not None:
        args_has_policy = args.get("has_policy", True)
        # If we are changing has_policy from True to False, avoid errors about
        # unused model parameters.
        if args_has_policy and not override_has_policy:
            load_state_dict_strict = False
        args["has_policy"] = override_has_policy
    if override_has_value is not None:
        args_has_value = args.get("has_value", True)
        # If we are changing has_value from True to False, avoid errors about
        # unused model parameters.
        if args_has_value and not override_has_value:
            load_state_dict_strict = False
        args["has_value"] = override_has_value

    cfg = conf.conf_cfgs.TrainTask(**args)
    model = new_model(cfg)

    # strip "module." prefix if model was saved with DistributedDataParallel wrapper
    state_dict = {
        (k[len("module.") :] if k.startswith("module.") else k): v
        for k, v in checkpoint["model"].items()
    }

    results = model.load_state_dict(state_dict, strict=load_state_dict_strict)
    if not load_state_dict_strict:
        # Even when not strict, still fail on keys that needed to be there but weren't.
        if len(results.missing_keys) > 0:
            raise RuntimeError(
                f"Missing keys in state dict when loading diplomacy_model: {results.missing_keys}"
            )
        if len(results.unexpected_keys) > 0:
            logging.info(
                f"This diplomacy_model supports outputs we aren't using, pruning extra keys: {results.unexpected_keys}"
            )

    model = model.to(map_location)

    if eval:
        model.eval()

    return model, args


def load_diplomacy_model_model(checkpoint_path, map_location="cpu", eval=False,) -> DiplomacyModel:
    """Load a diplomacy_model model.

    Args:
        checkpoint_path: File path to load from, e.g. my_model.ckpt
        map_location: Device string like "cpu" or "cuda" or "cuda:0" to load the model tensors on.
        eval: If true, set the model into inference mode.

    Returns:
        diplomacy_model_model
    """
    model, _args = load_diplomacy_model_model_and_args(
        checkpoint_path, map_location=map_location, eval=eval,
    )
    return model


@functools.lru_cache(maxsize=CACHE_SIZE)
def load_diplomacy_model_model_cached(
    *, checkpoint_path: str, map_location: str,
):
    """Load a diplomacy_model model in inference mode, with caching.

    Args:
        checkpoint_path: File path to load from, e.g. my_model.ckpt
        map_location: Device string like "cpu" or "cuda" or "cuda:0" to load the model tensors on.

    Returns:
        diplomacy_model_model
    """
    return load_diplomacy_model_model(checkpoint_path, map_location=map_location, eval=True,)
