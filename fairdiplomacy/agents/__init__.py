# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .base_agent import BaseAgent
from .model_sampled_agent import ModelSampledAgent
from .br_search_agent import BRSearchAgent
from .searchbot_agent import SearchBotAgent
from .ce1p_agent import CE1PAgent
from .fp1p_agent import FP1PAgent
from .random_agent import RandomAgent
from .repro_agent import ReproAgent


def build_agent_from_cfg(agent_stanza: "conf.conf_pb2.Agent") -> "fairdiplomacy.agents.BaseAgent":
    from google.protobuf.json_format import MessageToDict

    agent_name = agent_stanza.WhichOneof("agent")
    assert agent_name, f"Config must define an agent type: {agent_stanza}"
    agent_cfg = getattr(agent_stanza, agent_name)
    return {
        "br_search": BRSearchAgent,
        "model_sampled": ModelSampledAgent,
        "searchbot": SearchBotAgent,
        "ce1p": CE1PAgent,
        "fp1p": FP1PAgent,
        "repro": ReproAgent,
        None: None,
    }[agent_name](**MessageToDict(agent_cfg, preserving_proto_field_name=True))


__all__ = [
    build_agent_from_cfg,
    ModelSampledAgent,
    BRSearchAgent,
    SearchBotAgent,
    CE1PAgent,
    RandomAgent,
]
