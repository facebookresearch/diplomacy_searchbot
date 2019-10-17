# Installation

Clone the repo with submodules:
```
git clone --recursive git@github.com:fairinternal/fairdiplomacy.git
cd fairdiplomacy
```

Create a conda env:
```
conda create -y --name fairdiplomacy python=3.7
conda activate fairdiplomacy
conda install -y nodejs
pip install -e . -vv
```

Install singularity 3.x, or on FAIR cluster run:
```
module load singularity/3.4.1/gcc.7.3.0
```

# Running a Game

Run:
```
cd fairdiplomacy
python env.py
```

This will run the MILA supervised learning agent against six random agents, and save the results to `game.json`. The code [in env.py](https://github.com/fairinternal/fairdiplomacy/blob/master/fairdiplomacy/env.py#L74-L87) that implements this is shown here:

```
    env = Env(
        {
            "ITALY": MilaSLAgent(),
            "ENGLAND": RandomAgent(),
            "FRANCE": RandomAgent(),
            "GERMANY": RandomAgent(),
            "AUSTRIA": RandomAgent(),
            "RUSSIA": RandomAgent(),
            "TURKEY": RandomAgent(),
        }
    )
    results = env.process_all_turns()
    logging.info("Game over! Results: {}".format(results))
    env.save("game.json")
```

# Visualizing a Saved Game

Run:
```
./bin/open_visualizer.py
```

By default, this runs a webserver on `localhost:3000` (and a websocket server on `localhost:8432`)

If running on your devfair, be sure to run `ssh` with `-L 3000:localhost:3000 -L 8432:localhost:8432`

Log in as admin/password, and use "Load a game from disk" in the top-right to visualize a `game.json` file.

![Instructions for visualizing a game](https://github.com/diplomacy/diplomacy/blob/master/docs/images/visualize_game.png)

# What's included so far

An implementation of MILA's DipNet [https://arxiv.org/abs/1909.02128](https://arxiv.org/abs/1909.02128) is in progress at [models/dipnet/train_sl.py](https://github.com/fairinternal/fairdiplomacy/blob/master/fairdiplomacy/models/dipnet/train_sl.py)


# A Primer on the diplomacy.Game object

`game.get_state()` returns a dict containing the current board position. The most commonly accessed keys are:

`name`, returning the short-hand game phase:
```
>>> game.get_state()["name"]
'S1901M'
```

`units`, returning the locations of all units on the board:
```
>>> game.get_state()["units"]
{'AUSTRIA': ['A BUD', 'A VIE', 'F TRI'],
 'ENGLAND': ['F EDI', 'F LON', 'A LVP'],
 'FRANCE': ['F BRE', 'A MAR', 'A PAR'],
 'GERMANY': ['F KIE', 'A BER', 'A MUN'],
 'ITALY': ['F NAP', 'A ROM', 'A VEN'],
 'RUSSIA': ['A WAR', 'A MOS', 'F SEV', 'F STP/SC'],
 'TURKEY': ['F ANK', 'A CON', 'A SMY']}
```

`centers`, returning the supply centers controlled by each power:
```
>>> game.get_state()["centers"]
{'AUSTRIA': ['BUD', 'TRI', 'VIE'],
 'ENGLAND': ['EDI', 'LON', 'LVP'],
 'FRANCE': ['BRE', 'MAR', 'PAR'],
 'GERMANY': ['BER', 'KIE', 'MUN'],
 'ITALY': ['NAP', 'ROM', 'VEN'],
 'RUSSIA': ['MOS', 'SEV', 'STP', 'WAR'],
 'TURKEY': ['ANK', 'CON', 'SMY']}
```

`game.order_history` is a SortedDict of {short phase name => {power => [orders]}}
```
>>> game.order_history
{'S1901M': {'AUSTRIA': ['A VIE - GAL', 'F TRI H', 'A BUD - RUM'],
            'ENGLAND': ['F EDI - NTH', 'A LVP - YOR', 'F LON - ENG'],
            'FRANCE': ['F BRE - MAO', 'A PAR - BUR', 'A MAR S A PAR - BUR'],
            'GERMANY': ['F KIE - HOL', 'A BER - KIE', 'A MUN - BUR'],
            'ITALY': ['A VEN - PIE', 'A ROM - VEN', 'F NAP - ION'],
            'RUSSIA': ['A MOS - UKR',
                       'F STP/SC - BOT',
                       'A WAR - GAL',
                       'F SEV - BLA'],
            'TURKEY': ['A SMY - ARM', 'F ANK - BLA', 'A CON - BUL']},
 'F1901M': {'AUSTRIA': ['A VIE - GAL', 'F TRI H', 'A RUM S A ARM - SEV'],
 ...
```

`game.get_all_possible_orders()` returns a dict from location -> list of possible orders, e.g.
```
>>> game.get_all_possible_orders()
{'ADR': [],
 'AEG': [],
 'ALB': [],
 'ANK': ['F ANK S F SEV - ARM',
         'F ANK H',
         'F ANK S A CON',
         'F ANK - ARM',
         'F ANK S A SMY - CON',
         'F ANK S F SEV - BLA',
         'F ANK S A SMY - ARM',
         'F ANK - BLA',
         'F ANK - CON'],
  ...
```

`game.get_orderable_locations()` returns a map from power -> list of locations that need an order:
```
>>> game.get_orderable_locations()
{'AUSTRIA': ['BUD', 'TRI', 'VIE'],
 'ENGLAND': ['EDI', 'LON', 'LVP'],
 'FRANCE': ['BRE', 'MAR', 'PAR'],
 'GERMANY': ['BER', 'KIE', 'MUN'],
 'ITALY': ['NAP', 'ROM', 'VEN'],
 'RUSSIA': ['MOS', 'SEV', 'STP', 'WAR'],
 'TURKEY': ['ANK', 'CON', 'SMY']}
```

`game.set_orders(power, orders_list)` sets orders for that power's units, e.g.
```
>>> game.set_orders("TURKEY", ["F ANK - BLA", "A CON H"])
```

`game.process()` processes the orders that have been set, and moves to the next phase


Complete documentation is available [here](https://docs.diplomacy.ai/en/stable/api/diplomacy.engine.game.html)
