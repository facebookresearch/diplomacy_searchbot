# Playing Against a Bot

Run:
```
./bin/open_visualizer.py
```

By default, this runs a webserver on `localhost:3000` (and a websocket server on `localhost:8432`)

If running on your devfair, be sure to run `ssh` with `-L 3000:localhost:3000 -L 8432:localhost:8432`

Log in as admin/password, and use "Load a game from disk" in the top-right to visualize a `game.json` file.

![Instructions for visualizing a game](https://github.com/diplomacy/diplomacy/blob/master/docs/images/visualize_game.png)

Create a new standard game with 1 human user. Then run
```
python run.py --adhoc --cfg conf/c03_launch_bot/launch_bot.prototxt I.agent=agents/dipnet
```

To play against six CFR bots sharing two GPUs, run
```
python run.py --adhoc --cfg conf/c03_launch_bot/launch_bot.prototxt \
    I.agent=agents/cfr1p \
    agent.cfr1p.postman_sync_batches=False \
    reuse_model_servers=2
```
