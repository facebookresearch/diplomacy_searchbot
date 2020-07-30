from parlai.agents.transformer.transformer import TransformerGeneratorAgent
from parlai.core.loader import register_agent

from typing import List


SPECIAL_TOKS = [
    "__VEN__",
    "__ALB__",
    "__KIE__",
    "__-__",
    "__BAR__",
    "__NWG__",
    "__TUS__",
    "__EDI__",
    "__GRE__",
    "__PRU__",
    "__BUD__",
    "__HEL__",
    "__IRI__",
    "__SKA__",
    "__GAL__",
    "__TYS__",
    "__RUM__",
    "__NAP__",
    "__SMY__",
    "__LON__",
    "__ADR__",
    "__C__",
    "__BOH__",
    "__EAS__",
    "__BEL__",
    "__ANK__",
    "__MAR__",
    "__APU__",
    "__TUN__",
    "__PIE__",
    "__SPA/NC__",
    "__SPA__",
    "__HOL__",
    "__STP/SC__",
    "__SIL__",
    "__MUN__",
    "__BUL/SC__",
    "__YOR__",
    "__LYO__",
    "__ION__",
    "__TYR__",
    "__CON__",
    "__WES__",
    "__ENG__",
    "__B__",
    "__NAF__",
    "__UKR__",
    "__AEG__",
    "__SER__",
    "__ROM__",
    "__WAR__",
    "__D__",
    "__BUR__",
    "__VIA__",
    "__STP/NC__",
    "__VIE__",
    "__A__",
    "__BUL/EC__",
    "__R__",
    "__LVP__",
    "__GAS__",
    "__BAL__",
    "__SPA/SC__",
    "__BUL__",
    "__BLA__",
    "__F__",
    "__TRI__",
    "__ARM__",
    "__SWE__",
    "__H__",
    "__RUH__",
    "__NTH__",
    "__NWY__",
    "__BOT__",
    "__DEN__",
    "__NAO__",
    "__WAL__",
    "__BER__",
    "__PIC__",
    "__MOS__",
    "__STP__",
    "__BRE__",
    "__PAR__",
    "__S__",
    "__SEV__",
    "__MAO__",
    "__SYR__",
    "__FIN__",
    "__LVN__",
    "__CLY__",
    "__POR__",
    "__[EO_O]__",
    "__[EO_STATE]__",
    "__[EO_M]__",
]


class DiplomacyTokMixin:
    """
    Mixin object to add special tokens for diplomacy.
    """

    def _get_special_tokens(self) -> List[str]:
        return SPECIAL_TOKS


@register_agent("special_tok_generator")
class SpecialTokGeneratorAgent(DiplomacyTokMixin, TransformerGeneratorAgent):
    """
    Special token TransformerGeneratorAgent.

    Identical to the transformer/generator agent in ParlAI except that it
    adds all of the special tokens from diplomacy.
    """

    pass
