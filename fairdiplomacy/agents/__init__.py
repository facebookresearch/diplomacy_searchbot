# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import heyhi
from .base_agent import BaseAgent
from .model_sampled_agent import ModelSampledAgent
from .br_search_agent import BRSearchAgent
from .searchbot_agent import SearchBotAgent
from .ce1p_agent import CE1PAgent
from .random_agent import RandomAgent
from .repro_agent import ReproAgent


AGENT_CLASSES = {
    "br_search": BRSearchAgent,
    "model_sampled": ModelSampledAgent,
    "searchbot": SearchBotAgent,
    "ce1p": CE1PAgent,
    "repro": ReproAgent,
    "random": RandomAgent,
}


def build_agent_from_cfg(
    agent_stanza: "conf.agents_cfgs.Agent", **redefines
) -> "fairdiplomacy.agents.BaseAgent":
    assert agent_stanza.agent is not None, f"Config must define an agent type: {agent_stanza}"
    agent_cfg = agent_stanza.agent

    # FIXME(akhti): this code turned out to have some issue
    #  - changes should happend on config load, otherwise config composition is
    #    broken.
    #  - old flags should be deleted on migration
    #  - if both old and new flags are present, old flag has a priority
    # If you want to use this code, please, ping akhti to fix the issue first.
    # for k_from, k_to in CFG_TRANSLATE.get(agent_name, {}).items():
    #     if heyhi.conf_is_set(agent_cfg, k_from):
    #         v = heyhi.conf_get(agent_cfg, k_from)
    #         heyhi.conf_set(agent_cfg, k_to, v)

    if redefines:
        agent_cfg = agent_cfg.to_editable()
        # handle redefines
        for k, v in redefines.items():
            heyhi.conf_set(agent_cfg, k, v)
        agent_cfg = agent_cfg.to_frozen()

    return AGENT_CLASSES[agent_stanza.which_agent](agent_cfg)
