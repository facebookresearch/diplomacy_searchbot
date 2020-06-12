from .base_agent import BaseAgent
from .dipnet_agent import DipnetAgent
from .mila_sl_agent import MilaSLAgent
from .br_search_agent import BRSearchAgent
from .cfr1p_agent import CFR1PAgent
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
        "mila": MilaSLAgent,
        "br_search": BRSearchAgent,
        "dipnet": DipnetAgent,
        "cfr1p": CFR1PAgent,
        "ce1p": CE1PAgent,
        "fp1p": FP1PAgent,
        "repro": ReproAgent,
        None: None,
    }[agent_name](**MessageToDict(agent_cfg, preserving_proto_field_name=True))


__all__ = [build_agent_from_cfg, DipnetAgent, MilaSLAgent, BRSearchAgent, CFR1PAgent, CE1PAgent, RandomAgent]
