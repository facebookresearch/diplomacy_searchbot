# Training exploit agent

Exploit agent learns to play against a fixed supervised agent. Configs for
exploit agents are in [conf/c04_exploit/](conf/c04_exploit/). Take the latest
and run:
```
python run.py --adhoc --cfg conf/c04_exploit/exploit_??.prototxt I.launcher=slurm_8gpus
```

The most important metric is `score/is_clear_win`, i.e., the win rate. It should be above 0.2 after half an hour minutes (12 epochs) and above 0.5 after an hour.
