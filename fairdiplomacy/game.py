import diplomacy


class Game(diplomacy.Game):
    def __init__(self, **kwargs):
        if "rules" not in kwargs:
            kwargs["rules"] = [
                "NO_DEADLINE",
                "CD_DUMMIES",
                "ALWAYS_WAIT",
                "SOLITAIRE",
                "NO_PRESS",
                "POWER_CHOICE",
            ]
        super().__init__(**kwargs)

    @property
    def is_game_done(self):
        return super().is_game_done or int(self.phase.split()[1]) >= 1935

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


def sort_phase_key(phase):
    return (
        int(phase[1:5]),
        {"S": 0, "F": 1, "W": 2}[phase[0]],
        {"M": 0, "R": 1, "A": 2}[phase[5]],
    )
