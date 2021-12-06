#!/bin/bash

cd $(dirname $0)/..
mkdir -p models
wget https://github.com/facebookresearch/diplomacy_searchbot/releases/download/1.0/blueprint.pt -O blueprint.pt
wget https://github.com/facebookresearch/diplomacy_searchbot/releases/download/1.1/neurips21_dora_epoch006000.ckpt -O models/neurips21_dora_epoch006000.ckpt
wget https://github.com/facebookresearch/diplomacy_searchbot/releases/download/1.1/neurips21_dora_value_epoch006000.ckpt -O models/neurips21_dora_value_epoch006000.ckpt
wget https://github.com/facebookresearch/diplomacy_searchbot/releases/download/1.1/neurips21_human_dnvi_npu_epoch000500.ckpt -O models/neurips21_human_dnvi_npu_epoch000500.ckpt
wget https://github.com/facebookresearch/diplomacy_searchbot/releases/download/1.1/neurips21_human_dnvi_npu_value_epoch000500.ckpt -O models/neurips21_human_dnvi_npu_value_epoch000500.ckpt
wget https://github.com/facebookresearch/diplomacy_searchbot/releases/download/1.1/neurips21_supervised.ckpt -O models/neurips21_supervised.ckpt
wget https://github.com/facebookresearch/diplomacy_searchbot/releases/download/1.1/neurips21_fva_dora_epoch007000.ckpt -O models/neurips21_fva_dora_epoch007000.ckpt
wget https://github.com/facebookresearch/diplomacy_searchbot/releases/download/1.1/neurips21_fva_dora_value_epoch007000.ckpt -O models/neurips21_fva_dora_value_epoch007000.ckpt
wget https://github.com/facebookresearch/diplomacy_searchbot/releases/download/1.1/neurips21_supervised_heavy.ckpt -O models/neurips21_supervised_heavy.ckpt
