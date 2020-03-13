from .base_agent import BaseAgent
from .dipnet_agent import DipnetAgent
from .mila_sl_agent import MilaSLAgent
from .br_search_agent import BRSearchAgent
from .cfr1p_agent import CFR1PAgent
from .random_agent import RandomAgent


def build_agent_from_cfg(agent_stanza: "conf.conf_pb2.Agent") -> "fairdiplomacy.agents.BaseAgent":
    from google.protobuf.json_format import MessageToDict

    agent_name = agent_stanza.WhichOneof("agent")
    assert agent_name, f"Config must define an agent type: {agent_stanza}"
    agent_cfg = getattr(agent_stanza, agent_name)
    return {
        "mila": MilaSLAgent,
        "br_search": BRSearchAgent,
        "dipnet": DipnetAgent,
        "cfr1p": CFR1PAgent,
        None: None,
    }[agent_name](**MessageToDict(agent_cfg, preserving_proto_field_name=True))


__all__ = [build_agent_from_cfg, DipnetAgent, MilaSLAgent, BRSearchAgent, CFR1PAgent, RandomAgent]
