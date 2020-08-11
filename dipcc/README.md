## Overview

`dipcc` is a Diplomacy game engine written from scratch in c++. It aims to be a faster alternative to the [MILA Diplomacy Engine](https://github.com/diplomacy/diplomacy) with a similar API.

`dipcc` also provides a python interface `pydipcc` via [pybind11](https://github.com/pybind/pybind11). The ground truth python API is at [dipcc/pybind/pybind.cc](dipcc/dipcc/pybind/pybind.cc)

## Build

Run `./compile.sh`, or from the repo root run `make dipcc`

To compile with verbose adjudicator logging, run `MODE=DEBUG ./compile.sh`

## JSON Encoding/Decoding

The biggest API change from the MILA engine to `dipcc` is that the json encoding/decoding functions operate on strings, and the json processing is done by `dipcc`.

New `dipcc` API:

```
with open("/path/to/game.json", "r") as f:
    s = f.read()  # string

game = pydipcc.Game.from_json(s)

json_string = game.to_json()
```

Old API:

```
with open("/path/to/game.json", "r") as f:
    d = json.load(f)  # python dict

game = diplomacy.Game.from_saved_game_format(d)

python_dict = diplomacy.utils.export.to_saved_game_format(game) 
```

## pydipcc.ThreadPool

A c++ thread pool allows for parallelizing some of the slower methods of `dipcc` when operating concurrently on multiple `Game` objects. 
The ground truth python API is at [dipcc/pybind/pybind.cc](dipcc/dipcc/pybind/pybind.cc)

The main methods are:

`pydipcc.ThreadPool.process_multi(List[Game])`: calls `game.process()` on each of the games passed to it

`pydipcc.ThreadPool.encode_inputs_multi(List[Game], <data buffers>)`: more easily called from `encode_batch_inputs(ThreadPool, List[Game]) -> List[DataFields]` in [dipnet_agent.py](https://github.com/fairinternal/fairdiplomacy/blob/f2919a5d07e7d498e8af4e399cbb571305b9b565/fairdiplomacy/agents/dipnet_agent.py#L134),
this produces the tensors that are provided to the no-text pytorch model.

`pydipcc.ThreadPool.decode_order_idxs(order_idxs)`: converts a `LongTensor` of order vocabulary idxs to their corresponding order strings. 

The primary example usage of these functions is in [threaded_search_agent.py](fairdiplomacy/agents/threaded_search_agent.py)
