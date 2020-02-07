# Installation

Clone the repo with submodules:
```
git clone --recursive git@github.com:fairinternal/fairdiplomacy.git
cd fairdiplomacy
```

The following command will create/activate conda env with all needed modules:
```
. fair_activate.sh
```

Or you can do this manually:
```
conda create -y --name fairdiplomacy python=3.7
conda activate fairdiplomacy
conda install -y nodejs
pip install -e . -vv
pip install -e ./thirdparty/github/fairinternal/postman/nest/
pip install /checkpoint/hnr/wheels/postman-0.1.1-cp37-cp37m-linux_x86_64.whl
```

Install singularity 3.x, or on FAIR cluster run:
```
module load singularity/3.4.1/gcc.7.3.0
```

If you are getting warnings about `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION`, you may also need to set this environment variable:
```
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp
```

# Training a Model

The model code is in [fairdiplomacy/models/dipnet/](fairdiplomacy/models/dipnet/). The model architecture is defined in [dipnet.py](fairdiplomacy/models/dipnet/dipnet.py)

The script to train a model is in [fairdiplomacy/bin/train_sl.py](fairdiplomacy/bin/train_sl.py)

To train a model on the cluster, see the scripts in [slurm/](slurm/), specifically [example_train_sl.sh](slurm/example_train_sl.sh).


# Comparing Agents

See [fairdiplomacy/bin/compare_agents.py](fairdiplomacy/bin/compare_agents.py)

e.g. to compare a trained dipnet model to the mila bot, run:
```
python fairdiplomacy/bin/compare_agents.py /path/to/dipnet.pth mila ITALY --seed 0 -o output.json
```

This plays a game with one dipnet bot (playing Italy) vs. 6 mila bots, and writes the output to `output.json`

To run a full comparison suite on the cluster see [slurm/compare_agents.sh](slurm/compare_agents.sh)


# Visualizing a Saved Game

Run:
```
./bin/open_visualizer.py
```

By default, this runs a webserver on `localhost:3000` (and a websocket server on `localhost:8432`)

If running on your devfair, be sure to run `ssh` with `-L 3000:localhost:3000 -L 8432:localhost:8432`

Log in as admin/password, and use "Load a game from disk" in the top-right to visualize a `game.json` file.

![Instructions for visualizing a game](https://github.com/diplomacy/diplomacy/blob/master/docs/images/visualize_game.png)


# Playing Against the MILA Bot

After [opening the visualizer](#visualizing-a-saved-game), create a new standard game with 1 human user. Then run
```
python thirdparty/github/diplomacy/research/diplomacy_research/scripts/launch_bot.py
```

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
