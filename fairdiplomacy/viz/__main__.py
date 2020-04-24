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

import flask

import diplomacy.engine.renderer
import fairdiplomacy.game


app = flask.Flask("viz", root_path=pathlib.Path(__file__).parent)
manager: Optional["TBManager"] = None

TMP_PATH = pathlib.Path("/tmp/tb_links")
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
            Phase: {{ phase }} / {{ num_phases }}
                {% if phase > 0 %}<a href="?game={{ game_json_path }}&phase={{ phase - 1 }}">prev</a>{% else %}prev{% endif %}
                {% if phase < num_phases - 1 %}<a href="?game={{ game_json_path }}&phase={{ phase + 1 }}">next</a>{% endif %}
            {% endif %}
    </nav>
    {% if image %}
    <div>
        <div style="margin: 0 auto; width: 918px">
            {{ image|safe }}
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
    {% endif %}
</div>
</body>
</html>
"""


@app.route("/")
def root():
    game_json_path = flask.request.args.get("game")
    if not game_json_path:
        return flask.render_template_string(INDEX_HTML, DEMO_QUERY=DEMO_QUERY)
    try:
        phase = int(flask.request.args.get("phase") or 0)
    except ValueError:
        return "Bad phase! Check url"
    game_json_path = pathlib.Path(game_json_path)
    if not game_json_path.exists():
        return f"Bad game.json (not found): {game_json_path}"
    with game_json_path.open() as stream:
        game_json = json.load(stream)
    num_phases = len(game_json["phases"])
    phase = min(phase, num_phases - 1)
    del game_json["phases"][phase + 1 :]
    game = fairdiplomacy.game.Game.from_saved_game_format(game_json)
    image = diplomacy.engine.renderer.Renderer(game).render()
    return flask.render_template_string(INDEX_HTML, **locals())


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--port",
        type=int,
        default=8894,
        help="Where to server toplog. This port has to be forwarded.",
    )
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
