# Installation

Clone the repo with submodules:
```
git clone git@github.com:fairinternal/fairdiplomacy.git --recursive
cd fairdiplomacy
```

Create a conda env:
```
conda create --name fairdiplomacy python=3.7
conda activate fairdiplomacy
conda install nodejs
pip install -e . -vv
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

By default, this runs a webserver on `localhost:3000`. I (jgray) run this on my laptop, but we could probably find a port-forwarding solution that would allow this to be run on devfair. Log in as admin/password, and use "Load a game from disk" in the top-right to visualize a `game.json` file.

![Instructions for visualizing a game](https://github.com/diplomacy/diplomacy/blob/master/docs/images/visualize_game.png)

# What's included so far

An implementation of MILA's DipNet [https://arxiv.org/abs/1909.02128](https://arxiv.org/abs/1909.02128) is in progress at [models/dipnet/train_sl.py](https://github.com/fairinternal/fairdiplomacy/blob/master/fairdiplomacy/models/dipnet/train_sl.py)
