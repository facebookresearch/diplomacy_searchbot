"""A simple web server to show save game.json's.

Usage:
    pip install flask
    nohup python -m fairdiplomacy.viz --port 8894 &

Forward port 8894 and open http://localhost:8894 in the browser.
It's recommended to NOT change the port so that it's easier to share links to games among team members.
"""

from typing import Optional
import argparse
import json
import pathlib
import urllib.parse

import flask
import pandas as pd

import fairdiplomacy.game
from fairdiplomacy.models.consts import POWERS
import fairdiplomacy.utils.game_scoring

TESTCASE_PATH = pathlib.Path(__file__).parent.parent.parent / "test_situations.json"

app = flask.Flask("viz", root_path=pathlib.Path(__file__).parent)

DEMO_QUERY = "?game=/checkpoint/yolo/fairdiplomacy/data/20200420_noam_cfr_selfplay_games_more/games/game_AUS.100.json"

INDEX_HTML = """
<!doctype html>
<html>
<head profile="http://www.w3.org/2005/10/profile">
<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/css/bootstrap.min.css" integrity="sha384-Vkoo8x4CGsO3+Hhxv8T/Q5PaXtkKtu6ug5TOeNV6gBiFeWPGFN9MuhOf23Q9Ifjh" crossorigin="anonymous">
<link rel="icon" type="image/ico" href="/favicon.ico" />
</head>
<body>
<div class="container">
    <nav class="navbar navbar-light" style="background-color: #e3f2fd;">
            <a href="?" class="navbar-brand mb-0 h1">Diplomacy viz</a>
            {% if num_phases %}
            Phase: {{ phase_id + 1 }} / {{ num_phases }} ({{phase}})
                {% if first_phase %}<a href="?phase={{ first_phase }}&game={{ game_json_path }}{{ url_suffix }}">first</a>{% else %}first{% endif %}
                {% if prev_phase %}<a href="?phase={{ prev_phase }}&game={{ game_json_path }}{{ url_suffix }}">prev</a>{% else %}prev{% endif %}
                {% if next_phase %}<a href="?phase={{ next_phase }}&game={{ game_json_path }}{{ url_suffix }}">next</a>{% else %}next{% endif %}
                {% if last_phase %}<a href="?phase={{ last_phase }}&game={{ game_json_path }}{{ url_suffix }}">last</a>{% else %}last{% endif %}
            {% endif %}
    </nav>
    {% if image %}
    {% if test_situation %}
    <pre>{{ test_situation }}</pre>
    {% endif %}
    <div>
        <div style="margin: 0 auto; width: 918px">
            {{ image|safe }}
        <p>
        <pre>{{ game_scores }}</pre>
        </p>
        </div>
    </div>
    {% else %} 
    <div class="mt-4">
        <form action="." class="form-inline">
            <input name="game" class="form-control col-sm-10" placeholder="path to the game.json " />
            <input type="submit" class="btn btn-primary" value="Show the game" />
        </form>
    </div>
    <small>Note: you can also pass the path in url, e.g., <a href="{{DEMO_QUERY}}">{{DEMO_QUERY}}.</a>
    <h4>Load test situation</h4>
    <ul>
        {% for name in test_situations %}
        <li><a href="/test/{{name}}/">{{name}}</a></li>
        {% endfor %}
    </ul>
    {% endif %}
</div>
</body>
</html>
"""


def load_test_situations():
    assert TESTCASE_PATH.exists(), TESTCASE_PATH
    with TESTCASE_PATH.open() as stream:
        return json.load(stream)


def maybe_load_game(game_json_path: str) -> Optional[fairdiplomacy.game.Game]:
    game_json_path = pathlib.Path(game_json_path)
    if not game_json_path.exists():
        return None
    with game_json_path.open() as stream:
        game_json = json.load(stream)
    game = fairdiplomacy.game.Game.from_saved_game_format(game_json)
    return game


@app.route("/")
def root():
    game_json_path = flask.request.args.get("game")
    if not game_json_path:
        return flask.render_template_string(
            INDEX_HTML, DEMO_QUERY=DEMO_QUERY, test_situations=list(load_test_situations())
        )
    phase = flask.request.args.get("phase") or "S1901M"
    game = maybe_load_game(game_json_path)
    if game is None:
        return f"Bad game.json (not found): {game_json_path}"
    phase_list = list(game.order_history)
    if game.get_state()["name"] not in phase_list:
        phase_list += [game.get_state()["name"]]
    try:
        phase_id = phase_list.index(phase)
    except ValueError:
        return "Bad phase. Known: " + " ".join(
            f"<a href='?game={game_json_path}&phase={phase}'>{phase}</a>" for phase in phase_list
        )
    num_phases = len(phase_list)
    prev_phase = phase_list[phase_id - 1] if phase_id > 0 else ""
    next_phase = phase_list[phase_id + 1] if phase_id < len(phase_list) - 1 else ""
    first_phase = phase_list[0] if phase_id > 0 else ""
    last_phase = phase_list[-1] if phase_id < len(phase_list) - 1 else ""

    test_situation_name = flask.request.args.get("test")
    if test_situation_name:
        test_situation = json.dumps(load_test_situations()[test_situation_name], indent=2)
        url_suffix = f"&test={test_situation_name}"
    else:
        url_suffix = ""

    game = fairdiplomacy.game.Game.clone_from(game, up_to_phase=phase)
    game_scores = {
        p: fairdiplomacy.utils.game_scoring.compute_game_scores(
            i, game.to_saved_game_format()
        )._asdict()
        for i, p in enumerate(POWERS)
    }
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    game_scores = str(pd.DataFrame(game_scores))
    image = game.render(incl_abbrev=True)
    return flask.render_template_string(INDEX_HTML, **locals())


@app.route("/test/<test_situation_name>/")
def show_test_situation(test_situation_name):
    test_situations = load_test_situations()
    if test_situation_name not in test_situations:
        known = " ".join(sorted(test_situations))
        return f"Uknown situation. Known: {known}"

    situation = test_situations[test_situation_name]
    game = maybe_load_game(situation["game_path"])
    if game is None:
        return f"Bad game.json (not found): " + situation["game_path"]

    return flask.redirect(
        "/?"
        + urllib.parse.urlencode(
            dict(game=situation["game_path"], phase=situation["phase"], test=test_situation_name)
        ),
        code=307,
    )


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--port",
        type=int,
        default=8894,
        help="Where to server toplog. This port has to be forwarded.",
    )
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port, threaded=True, debug=True)


if __name__ == "__main__":
    main()
