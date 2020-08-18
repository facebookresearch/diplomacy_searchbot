import diplomacy

POWERS = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]


class Game(diplomacy.Game):
    def __init__(self, max_year=1935, **kwargs):
        self.max_year = max_year
        if "rules" not in kwargs:
            kwargs["rules"] = [
                "NO_DEADLINE",
                "CD_DUMMIES",
                "ALWAYS_WAIT",
                "SOLITAIRE",
                "NO_PRESS",
                "POWER_CHOICE",
            ]
        if "ignore_multiple_disbands_per_unit" not in kwargs:
            kwargs["ignore_multiple_disbands_per_unit"] = True
        super().__init__(**kwargs)

    def get_phase_name(self, phase_idx):
        return str(list(self.state_history.keys())[phase_idx])

    @property
    def is_game_done(self):
        return super().is_game_done or int(self.phase.split()[1]) >= self.max_year

    def to_saved_game_format(self, *args, **kwargs):
        return diplomacy.utils.export.to_saved_game_format(self, *args, **kwargs)

    @classmethod
    def from_saved_game_format(cls, saved_game):
        """Copied from diplomacy.utils.export but using this Game class"""
        game_id = saved_game.get("id", None)
        kwargs = {
            diplomacy.utils.strings.MAP_NAME: saved_game.get("map", "standard"),
            diplomacy.utils.strings.RULES: saved_game.get("rules", []),
        }

        game = Game(game_id=game_id, **kwargs)

        if "phases" in saved_game:
            phase_history = []
            for phase_dct in saved_game.get("phases", []):
                phase_history.append(
                    diplomacy.utils.game_phase_data.GamePhaseData.from_dict(phase_dct)
                )
            game.set_phase_data(phase_history, clear_history=True)
            return game

        elif "order_history" in saved_game:
            for phase, phase_orders in sorted(
                saved_game["order_history"].items(), key=lambda kv: sort_phase_key(kv[0])
            ):
                assert phase == game._phase_abbr()
                for power, orders in phase_orders.items():
                    game.set_orders(power, orders)
                game.process()
            return game

    @classmethod
    def clone_from(cls, other, up_to_phase=None):
        kwargs = {
            "game_id": other.game_id,
            diplomacy.utils.strings.MAP_NAME: getattr(other, "map_name", "standard"),
            diplomacy.utils.strings.RULES: getattr(other, "rules", []),
        }
        game = Game(**kwargs)

        clone_phases = other.get_phase_history() + [other.get_phase_data()]
        if up_to_phase is not None:
            up_to_phase_key = sort_phase_key(up_to_phase)
            clone_phases = [p for p in clone_phases if sort_phase_key(p.name) <= up_to_phase_key]
        game.set_phase_data(clone_phases)

        return game


def sort_phase_key(phase):
    if phase == "COMPLETED":
        return (1e6,)
    else:
        return (
            int(phase[1:5]),
            {"S": 0, "F": 1, "W": 2}[phase[0]],
            {"M": 0, "R": 1, "A": 2}[phase[5]],
        )
