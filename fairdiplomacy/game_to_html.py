from typing import Optional
import argparse
import json
import pathlib
import urllib.parse

import pandas as pd
import jinja2

import fairdiplomacy.game


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
    </nav>
    {% if test_situation %}
    <pre>{{ test_situation }}</pre>
    {% endif %}
    <div>
    {%for i in range(num_phases)%}
    {% if i > 0 and i < num_phases - 1 and phase_names[i - 1][1:5] != phase_names[i][1:5] %}</div><div>{% endif %}
    <a href="#{{phase_names[i]}}">{{phase_names[i]}}</a>
    {% endfor %}
    </div>
    <p/>
    {% for i in range(num_phases) %}
    <div>
        <a id="{{phase_names[i]}}"/>
        {{ phase_names[i] }}
        <div style="margin: 0 auto; width: 918px">
            {{ images[i]|safe }}
        <p>
        </p>
        </div>
    </div>
    {% endfor %}
</div>
</body>
</html>
"""


def game_to_html(game):

    phase_names = [p.name for p in game.get_phase_history()]
    num_phases = len(phase_names)

    images = [
        fairdiplomacy.game.Game.clone_from(game, up_to_phase=phase).render(incl_abbrev=True)
        for phase in phase_names
    ]
    template = jinja2.Template(INDEX_HTML)
    return template.render(**locals())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path")
    parser.add_argument("-o", default="game.html")
    args = parser.parse_args()

    game = fairdiplomacy.game.Game.from_saved_game_format(json.load(open(args.json_path)))
    html = game_to_html(game)

    with open(args.o, "w") as f:
        f.write(html)


if __name__ == "__main__":
    main()
