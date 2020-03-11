#!/usr/bin/env python

import argparse
import logging
import torch
from pprint import pprint

from fairdiplomacy.models.dipnet.load_model import load_dipnet_model
from fairdiplomacy.models.dipnet.train_sl import validate


logging.basicConfig(format="%(asctime)s [%(levelname)s]: %(message)s", level=logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="/checkpoint/jsgray/diplomacy/dipnet.pth")
parser.add_argument("--data-cache", default="/checkpoint/jsgray/diplomacy/data_cache.pth")
args = parser.parse_args()


logging.info("Loading model")
model = load_dipnet_model(args.checkpoint, map_location="cuda", eval=True)
args = torch.load(args.checkpoint)["args"]

logging.info("Loading dataset")
_, val_set = torch.load(args.data_cache)

logging.info("Validating...")
policy_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
value_loss_fn = torch.nn.SmoothL1Loss(reduction="none")
pprint(
    validate(
        model, val_set, policy_loss_fn, value_loss_fn, args.batch_size, args.value_loss_weight
    )
)
