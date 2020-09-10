#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Script to increase bart size, should be run one-time to increase the position embedding size
to incorporate more context
Example usage
```
python increase_bart_size.py --model_save_path /private/home/wyshi/ParlAI/data/models/ \
--original_model_path /private/home/wyshi/ParlAI/data/models/bart/bart_large/
```
args:
--n_positions: default 2048
--model_save_path: the path to save the enlarged mode
--original_model_path: the path to the original model 
"""
import parlai.utils.pickle
import torch
import numpy as np
from parlai.utils.torch import atomic_save
from parlai_diplomacy.utils.param_sweeps.param_sweep import bash
from parlai.utils import logging
import os
import json
from datetime import datetime
import argparse

# seed
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_save_path", type=str,
    )
    parser.add_argument(
        "--original_model_path", type=str,
    )
    parser.add_argument(
        "--n_positions", type=int, default=2048,
    )
    args = parser.parse_args()

    # increase to
    final_position_size = args.n_positions
    extra_size = final_position_size - 1024

    # load
    states = torch.load(
        os.path.join(args.original_model_path, "model"), pickle_module=parlai.utils.pickle,
    )
    with open(os.path.join(args.original_model_path, "model.dict.opt"), "r") as fh:
        model_dict_opt = json.load(fh)

    with open(os.path.join(args.original_model_path, "model.opt"), "r") as fh:
        model_opt = json.load(fh)

    model_dict_opt["override"]["n_positions"] = final_position_size
    model_dict_opt["n_positions"] = final_position_size
    model_opt["n_positions"] = final_position_size

    # variables
    dtype = states["model"]["encoder.position_embeddings.weight"].dtype
    device = states["model"]["encoder.position_embeddings.weight"].device
    embedding_size = states["model"]["encoder.position_embeddings.weight"].shape[1]

    # extra position embeddings
    extra_encoder_pos_embedding_tensor = torch.empty(
        extra_size, embedding_size, dtype=dtype, device=device,
    ).normal_(0, embedding_size ** -0.5)
    extra_decoder_pos_embedding_tensor = torch.empty(
        extra_size, embedding_size, dtype=dtype, device=device,
    ).normal_(0, embedding_size ** -0.5)

    # final position embeddings
    increased_encoder_tensor = torch.cat(
        (
            states["model"]["encoder.position_embeddings.weight"],
            extra_encoder_pos_embedding_tensor,
        ),
        dim=0,
    )
    increased_decoder_tensor = torch.cat(
        (
            states["model"]["decoder.position_embeddings.weight"],
            extra_decoder_pos_embedding_tensor,
        ),
        dim=0,
    )

    states["model"]["encoder.position_embeddings.weight"] = increased_encoder_tensor
    states["model"]["decoder.position_embeddings.weight"] = increased_decoder_tensor

    # save
    built_save_path = os.path.join(args.model_save_path, f"bart_{final_position_size}")
    save_path = os.path.join(built_save_path, f"bart_{final_position_size}")
    bash(f"mkdir -p {save_path}")

    # .built
    version_info = [str(datetime.now()) + "\n", "v1.0\n"]
    with open(os.path.join(built_save_path, ".built"), "w") as fh:
        fh.writelines(version_info)

    # model
    logging.info(f"enlarged state_dict saved to {os.path.join(save_path, 'model')}")
    atomic_save(
        states, os.path.join(save_path, "model"),
    )

    # save opts
    with open(os.path.join(save_path, "model.dict.opt"), "w") as fh:
        json.dump(model_dict_opt, fh)

    with open(os.path.join(save_path, "model.opt"), "w") as fh:
        json.dump(model_opt, fh)

    bash(
        f"cp {os.path.join(args.original_model_path, 'model.dict')} {os.path.join(save_path, 'model.dict')}"
    )
