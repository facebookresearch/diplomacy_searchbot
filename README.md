# FAIR Diplomacy

### Game info
Diplomacy is a strategic board game set in 1914 Europe.
The board is divided into fifty-six land regions and nineteen sea regions.
Forty-two of the land regions are divided among the seven Great Powers of the game: Austria-Hungary, England, France, Germany, Italy, Russia, and Turkey.
The remaining fourteen land regions are neutral at the start of the game.

Each power controls some regions and some units.
The number of the units controlled depends on the number of the controlled key regions called Supply Centers (SCs).
Simply put, more SCs means more units.
The goal of the game is to control more than half of all SCs by moving units into these regions and convincing other players to support you.

You can find the full rules [here](https://en.wikibooks.org/wiki/Diplomacy/Rules).
To get the game's spirit, watch some [games with comments](https://www.youtube.com/watch?v=k04DTyEBeWw).
What's more, you can play the game online on [WebDiplomacy](https://webdiplomacy.net/) either against bots or humans.

<p align="center">
<img src="docs/images/webdiplomacy.gif" alt="Diplomacy text" />
</p>

### Game types

We consider two modifications of the game: No-Press and Press.

 * In **No-Press** (aka GunBoat) diplomacy no communication between parties is allowed. Therefore the players has to infer possible alliances from actions of other players.
 * In **Press** players can communicate with each other privately before making the order. Note, that orders are revealed simultaneously after the negotiation stage.


### Getting updates

Check [FAIR Diplomacy Research group](https://fb.workplace.com/groups/268868111083357/) for updates.


## Quick start

### Installation

Clone the repo with submodules:
```
git clone --recursive git@github.com:fairinternal/fairdiplomacy.git
cd fairdiplomacy
```

The following command will create/activate conda env with all needed modules:
```
. fair_activate.sh
```

In order to rebuild dependencies after update, run `./bin/install_deps.sh`.

### Running tasks

The code has a single entry point, `run.py`, that can be used to train a model, compare agents, profile them, etc.
We refer to this kind of activity as a task.
To specify which task to run and what parameters to use, we use configs.
Below an example of a config that is used to train an agent with imitation learning on human data:

```
train {
    dataset_params: {
        data_dir: "/checkpoint/alerer/fairdiplomacy/facebook_notext/games"
        value_decay_alpha: 0.9;
    }
    batch_size: 2500;
    lr: 0.001;
    lr_decay: 0.99;
    clip_grad_norm: 0.5
    checkpoint: "./checkpoint.pth";
    lstm_dropout: 0.1;
    encoder_dropout: 0.2;
    num_encoder_blocks: 8;
}
```

We use text [protobuf](https://developers.google.com/protocol-buffers/docs/proto#simple) format to specify the configs.
Each task has a schema, a formal description of what parameters are allowed in each config, e.g., [here's the definition](https://github.com/fairinternal/fairdiplomacy/blob/f89c5b67fa6e9889ed723f372166a90504b36a80/conf/conf.proto#L73-L208) for the train task above.

Protobufs could be confusing, but good news - you don't have to understand them to run tasks.
Instead, you need to find the config for your task and run it.
We describe all tasks in the next section.
Here is an example of how to launch training on human data:

```
python run.py --adhoc --cfg conf/c02_sup_train/sl.prototxt
```

You can override any config parameter in command line using argparse-like syntax:

```
python run.py --adhoc --cfg conf/c02_sup_train/sl.prototxt batch_size=200 --dataset_params.value_decay_alpha=1.0
```

Note that it's optional to use "--" in front of overrides.

Finally, to launch something on the cluster add `I.launcher=slurm` (single GPU) or `I.launcher=slurm_8gpus` (8 GPUs).
Check documentation for [HeyHi](heyhi/), the configuration library, for more details.


### Tasks overview

In general, all configs are stored in [conf/](conf/) folder and grouped by tasks.
You can find all possible arguments for all tasks in [conf/conf.proto](conf/conf.proto) file.
Below are the most important tasks:

 * Supervised training task for GunBoat. Configs in `c02_sup_train`, docs [here](docs/train_sup.md).
 * Making 2 agents play against each other for evaluation. Configs in `c01_ag_cmp`, docs [here](docs/compare_agents.md).
 * Training RL agent. Configs in `c04_exploit`, docs [here](docs/selfplay.md).
 * Playing against a bot. Configs in `c03_launch_bot`, docs [here](docs/launch_bot.md).
 * Testing whether CFR agent plays correctly in some test situations. Configs in `c06_situation_check`, test situations in [test_situations.json](test_situations.json).

## Going deeper

We use an in-house fast C++ implementation of the diplomacy environment.
See [here](docs/game_engine.md) for how to interact with it.

The games could be serialized as JSON files, e.g., our GunBoat human data and test situations use this format.
You can use [viz](docs/vizualization.md) tool to quickly see what's there.

The text related models are based on [ParlAI](https://github.com/facebookresearch/ParlAI/blob/master/README.md) framework. See docs [here](parlai_diplomacy/).

Code structure:

 * [fairdiplomacy](fairdiplomacy/) - datasets, agents, and trainers for GunBoat part.
 * [conf](conf/) - all the configs for fairdiplomacy/ part.
 * [parlai_diplomacy](parlai_diplomacy/) - code for handling Press games.


External links:

 * [PostMan](https://github.com/fairinternal/postman) is an in-house tensor-friendly RPC library. We are using it for search and RL.
 * [TopLog](https://github.com/fairinternal/toplog) is a simple tool to have on demand TensorBoard. Both supervised and RL trainers output TB logs.
 * [Mila paper](https://papers.nips.cc/paper/8697-no-press-diplomacy-modeling-multi-agent-gameplay.pdf) that our GunBoat game is based on.
 * [DM paper](https://arxiv.org/pdf/2006.04635.pdf). We use some model improvement from the paper.
 * [Our workplace group](https://fb.workplace.com/groups/268868111083357)

## Contributing

We use [black](https://github.com/psf/black) auto-formatter to format all python code.

We use `clang-format-8 conf/*.proto -i` to autoformat proto.